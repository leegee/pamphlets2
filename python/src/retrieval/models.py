from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]

# Event IDs are never expected to use this value. It is used only to pad
# fixed-width ANN result rows when a filtered year contains fewer than k
# observations.
INVALID_EVENT_ID = np.iinfo(np.uint64).max


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """
    Logical constraints on an observation search.

    years:
        None for all years, or an inclusive ``(start, end)`` range.

    scale:
        None for all scales, or a tuple containing one or more scales.

    The space describes the global corpus region being analysed. Per-query
    restrictions such as "same publication year as this seed" are applied
    by the observation index at search time.
    """

    years: tuple[int, int] | None
    scale: tuple[str, ...] | None

    _VALID_SCALES = frozenset({
        "local",
        "medium",
        "broad",
    })

    def __post_init__(self) -> None:
        if self.years is not None:
            if isinstance(self.years, int):
                object.__setattr__(
                    self,
                    "years",
                    (self.years, self.years),
                )
            elif isinstance(self.years, tuple):
                if len(self.years) != 2:
                    raise ValueError( f"year range must contain exactly two years not {self.years}" )

                if not all(
                    isinstance(year, int)
                    for year in self.years
                ):
                    raise TypeError( "year range must contain integers" )

                start, end = self.years

                if start > end:
                    raise ValueError( "year range must be in ascending order" )
            else:
                raise TypeError( "years must be an int or a two-year tuple" )

        if self.scale is not None:
            if isinstance(self.scale, str):
                if self.scale not in self._VALID_SCALES:
                    raise ValueError( f"invalid scales: {[self.scale]}" )

                object.__setattr__(
                    self,
                    "scale",
                    (self.scale,),
                )
            elif isinstance(self.scale, tuple):
                if not self.scale:
                    raise ValueError( "scale selection must contain at least one scale" )

                if not all(
                    isinstance(scale, str)
                    for scale in self.scale
                ):
                    raise TypeError( "scale selection must contain strings" )

                invalid = set(self.scale) - self._VALID_SCALES

                if invalid:
                    raise ValueError( f"invalid scales: {sorted(invalid)}" )
            else:
                raise TypeError( "scale must be a string or tuple of strings" )


    def buckets(
        self,
        bucket_size: int = 50,
        direction: Literal["forward", "backward"] = "forward",
    ) -> Iterator[tuple[int, int]]:
        """
        Yield chronological year ranges within this search space.

        ``forward`` traverses from the earliest requested year to the latest.
        ``backward`` traverses from the latest requested year to the earliest.

        Failure mode:
            An unbounded space cannot be traversed because there is no
            finite chronological interval to divide into buckets.
        """
        if self.years is None:
            raise ValueError( "bucket traversal requires an explicit year range" )

        if not isinstance(bucket_size, int) or bucket_size <= 0:
            raise ValueError( "bucket_size must be a positive integer" )

        if direction not in ("forward", "backward"):
            raise ValueError( f"invalid direction: {direction!r}" )

        start, end = self.years

        if direction == "forward":
            current = start

            while current <= end:
                bucket_end = min(
                    current + bucket_size - 1,
                    end,
                )

                yield current, bucket_end
                current = bucket_end + 1

        else:
            current = end

            while current >= start:
                bucket_start = max(
                    start,
                    current - bucket_size + 1,
                )

                yield bucket_start, current
                current = bucket_start - 1

    def resolve_years(
        self,
        available_years: set[int] | frozenset[int],
    ) -> tuple[int, ...]:
        """
        Resolve the logical year constraint against available data.

        Failure mode:
            Unavailable years are silently excluded because they cannot
            contribute seed observations to the search.
        """
        resolved = set(available_years)

        if self.years is not None:
            start, end = self.years
            resolved = {
                year
                for year in resolved
                if start <= year <= end
            }

        return tuple(sorted(resolved))

    def resolve_scales(
        self,
        available_scales: set[str] | frozenset[str],
    ) -> tuple[str, ...]:
        """
        Resolve the logical scale constraint against available indexes.

        Failure mode:
            Requested scales without a corresponding search index are
            excluded rather than causing an unrelated scale to be used.

        TODO Reference the SCALES constant wherever it now lives
        """
        available = set(available_scales)

        if self.scale is None:
            return tuple(
                scale
                for scale in (
                    "local",
                    "medium",
                    "broad",
                )
                if scale in available
            )

        return tuple(
            scale
            for scale in self.scale
            if scale in available
        )


@dataclass(slots=True)
class SearchResult:
    """ANN results expressed in stable observation IDs."""

    event_ids: UInt64Array
    distances: Float32Array

    def __post_init__(self) -> None:
        if self.event_ids.shape != self.distances.shape:
            raise ValueError( "event_ids and distances must have identical shapes" )


@dataclass(slots=True)
class BatchSearchResult:
    """ANN results for multiple queries."""

    event_ids: UInt64Array
    distances: Float32Array

    def __post_init__(self) -> None:
        if self.event_ids.ndim != 2:
            raise ValueError( "batch event_ids must be two-dimensional" )

        if self.distances.ndim != 2:
            raise ValueError( "batch distances must be two-dimensional" )

        if self.event_ids.shape != self.distances.shape:
            raise ValueError( "batch event_ids and distances must have identical shapes" )

    def row(self, index: int) -> SearchResult:
        return SearchResult(
            event_ids=self.event_ids[index],
            distances=self.distances[index],
        )
