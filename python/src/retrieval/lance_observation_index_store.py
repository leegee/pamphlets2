from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Literal

import lancedb
import numpy as np

from lib.corpus_logging import logger
from retrieval.models import SCALES
from lib.corpus_config import (
    CORPUS_MAX_YEAR,
    CORPUS_MIN_YEAR,
    LANCE_INDEXES_DIR,
)

from .lance_observation_index import LanceObservationIndex
from .models import Float32Array, SearchResult, SearchSpace
from .observation_index import ObservationIndex
from .observation_index_store import ObservationIndexStore


_TABLE_PATTERN = re.compile(
    r"^(?P<scale>local|medium|broad)"
    r"__(?P<model>[^_]+(?:_[^_]+)*)"
    r"__(?P<year_start>\d{4})_(?P<year_end>\d{4})$"
)


class LanceObservationIndexStore(ObservationIndexStore):
    """
    ObservationIndexStore backed by chronologically partitioned Lance tables.

    Each physical table represents one embedding scale, one embedding model,
    and one chronological bucket. A logical SearchSpace may therefore map
    to several physical tables.
    """

    def __init__(
        self,
        lance_root: str | Path = LANCE_INDEXES_DIR,
        *,
        available_years=range(
            CORPUS_MIN_YEAR,
            CORPUS_MAX_YEAR + 1,
        ),
        available_scales: tuple[str, ...] = SCALES,
        dimensions: int = 768,
        nprobes: int = 20,
        model: str | None = None,
    ) -> None:
        self._lance_root = Path(lance_root)
        self._available_years = {
            int(year)
            for year in available_years
        }
        self._available_scales = tuple(
            available_scales
        )
        self._dimensions = dimensions
        self._nprobes = nprobes
        self._model = model

        self._db = lancedb.connect( str(self._lance_root) )

        self._tables = self._discover_tables()

        logger.info(
            "[retrieval] opened %d chronological Lance observation indexes",
            len(self._tables),
        )

    @property
    def available_scales(self) -> tuple[str, ...]:
        return self._available_scales

    def get(
        self,
        space: SearchSpace,
    ) -> dict[str, ObservationIndex]:
        scales = tuple(
            space.resolve_scales(
                set(self._available_scales)
            )
        )

        if not scales:
            raise ValueError(
                "SearchSpace resolves to no available scales"
            )

        years = tuple(
            space.resolve_years(
                self._available_years
            )
        )

        if not years:
            raise ValueError(
                "SearchSpace resolves to no available years"
            )

        year_start = min(years)
        year_end = max(years)

        indexes = {}

        for scale in scales:
            tables = self._tables_for_search(
                scale=scale,
                year_start=year_start,
                year_end=year_end,
            )

            if not tables:
                raise ValueError(
                    f"No Lance tables cover scale={scale} "
                    f"years={year_start}-{year_end}"
                )

            indexes[scale] = LanceObservationIndex(
                tables,
                dimensions=self._dimensions,
                year_start=year_start,
                year_end=year_end,
                model=self._model,
                nprobes=self._nprobes,
            )

        return indexes


    def diachronic_search(
        self,
        queries_by_scale: dict[str, Float32Array],
        space: SearchSpace,
        *,
        k: int,
        direction: Literal["forward", "backward"] = "forward",
    ) -> Iterator[
        tuple[
            tuple[int, int],
            dict[str, SearchResult],
        ]
    ]:
        """Search each physical chronological bucket independently.

        Each scale is searched independently within the current bucket. The
        retriever is responsible for merging those scale results and applying the
        final k limit across scales.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        if direction not in ("forward", "backward"):
            raise ValueError(
                f"invalid direction: {direction!r}"
            )

        scales = space.resolve_scales(
            set(self._available_scales)
        )

        if not scales:
            raise ValueError(
                "SearchSpace resolves to no available scales"
            )

        missing = [
            scale
            for scale in scales
            if scale not in queries_by_scale
        ]

        if missing:
            raise ValueError(
                f"missing query vectors for scales: {missing}"
            )

        buckets = self._buckets_for_search(
            space,
            direction=direction,
        )

        if not buckets:
            raise ValueError(
                "SearchSpace resolves to no chronological Lance buckets"
            )

        for bucket_start, bucket_end in buckets:
            bucket_space = SearchSpace(
                years=(bucket_start, bucket_end),
                scale=scales,
            )

            indexes = self.get(bucket_space)

            results_by_scale: dict[str, SearchResult] = {}

            for scale in scales:
                results_by_scale[scale] = indexes[scale].search(
                    queries_by_scale[scale],
                    k=k,
                )

            yield (
                (bucket_start, bucket_end),
                results_by_scale,
            )

    def _buckets_for_search(
        self,
        space: SearchSpace,
        *,
        direction: Literal["forward", "backward"],
    ) -> tuple[tuple[int, int], ...]:
        """
        Resolve a logical SearchSpace to physical Lance bucket boundaries.

        Physical table boundaries are authoritative. This prevents a
        logical range such as 1476–1920 from being split at arbitrary
        boundaries that do not correspond to the stored ANN indexes.
        """
        scales = space.resolve_scales(
            set(self._available_scales)
        )

        if space.years is None:
            requested_start = None
            requested_end = None
        else:
            requested_start, requested_end = space.years

        buckets = set()

        for (
            table_scale,
            model,
            bucket_start,
            bucket_end,
        ) in self._tables:
            if table_scale not in scales:
                continue

            if (
                requested_start is not None
                and bucket_end < requested_start
            ):
                continue

            if (
                requested_end is not None
                and bucket_start > requested_end
            ):
                continue

            buckets.add(
                (
                    bucket_start,
                    bucket_end,
                )
            )

        ordered = sorted(buckets)

        if direction == "backward":
            ordered.reverse()

        return tuple(ordered)

    def _discover_tables(self):
        discovered = {}

        for table_name in self._db.list_tables().tables:
            parsed = self._parse_table_name(table_name)

            if parsed is None:
                continue

            scale = parsed["scale"]
            model = parsed["model"]
            year_start = parsed["year_start"]
            year_end = parsed["year_end"]

            if scale not in self._available_scales:
                continue

            if self._model is not None and model != self._model:
                continue

            key = (
                scale,
                model,
                year_start,
                year_end,
            )

            try:
                table = self._db.open_table(table_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not open Lance table {table_name!r}"
                ) from exc

            discovered[key] = table

        if not discovered:
            raise RuntimeError(
                f"No compatible chronological Lance tables found in "
                f"{self._lance_root}"
            )

        return discovered

    def _tables_for_search(
        self,
        *,
        scale: str,
        year_start: int,
        year_end: int,
    ):
        candidates = []

        for (
            table_scale,
            model,
            bucket_start,
            bucket_end,
        ), table in self._tables.items():
            if table_scale != scale:
                continue

            if bucket_end < year_start:
                continue

            if bucket_start > year_end:
                continue

            candidates.append(
                (
                    bucket_start,
                    bucket_end,
                    table,
                )
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        return tuple(
            table
            for _, _, table in candidates
        )

    @staticmethod
    def _parse_table_name(
        table_name: str,
    ):
        match = _TABLE_PATTERN.match(
            table_name
        )

        if match is None:
            return None

        return {
            "scale": match.group("scale"),
            "model": match.group("model"),
            "year_start": int(
                match.group("year_start")
            ),
            "year_end": int(
                match.group("year_end")
            ),
        }
