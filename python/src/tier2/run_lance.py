"""
Tier 2 command-line runner.

This module owns orchestration only.

PostgreSQL is authoritative for Tier 1 event identity and provenance.
Lance is authoritative for embedding geometry.
The retrieval algorithm itself lives in tier2.analysis.

Failure mode:
    Tier 2 must not silently compensate for a broken Tier 1
    PostgreSQL/Lance completeness invariant.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lib.corpus_config import (
    CONCEPT_SETS,
    CORPUS_TIER2_DB_PATH,
    LANCE_INDEXES_DIR,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SCALES, SearchSpace
from tier2.analysis import (
    BATCH_SIZE,
    K,
    OVERSAMPLE,
    RRF_K,
    iter_concept_batches,
    resolve_concept_positions,
)
from tier2.sqlite import write_tier2_sqlite


def _available_event_years(connection) -> tuple[int, ...]:
    """
    Discover years from the current Tier 1 event universe.

    PostgreSQL is authoritative here; Lance may still contain legacy
    orphaned observations outside the current Tier 1 universe.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT pub_year
            FROM events
            WHERE pub_year IS NOT NULL
            ORDER BY pub_year
            """
        )

        return tuple(
            int(row[0])
            for row in cursor.fetchall()
        )


def _year_range(
    from_year: int | None,
    to_year: int | None,
) -> tuple[int, int] | None:
    if from_year is None and to_year is None:
        return None

    if from_year is None:
        return to_year, to_year

    if to_year is None:
        return from_year, from_year

    return from_year, to_year


def _resolve_search_scope(
    search_space: SearchSpace,
    available_years: tuple[int, ...],
):
    available_years_set = {
        int(year)
        for year in available_years
    }

    if search_space.years is None:
        candidate_years = tuple(
            sorted(available_years_set)
        )
    else:
        start, end = search_space.years

        candidate_years = tuple(
            year
            for year in sorted(available_years_set)
            if start <= year <= end
        )

    scales = tuple(
        search_space.resolve_scales(
            set(SCALES)
        )
    )

    if not scales:
        raise ValueError(
            "SearchSpace resolves to no available scales"
        )

    if not candidate_years:
        logger.warning(
            "[tier2] SearchSpace resolves to no searchable years"
        )

    return candidate_years, scales



def _build_indexes_by_year(
    *,
    lance_root: str | Path,
    candidate_years: tuple[int, ...],
    scales: tuple[str, ...],
):
    """
    Build one logical Lance index set per publication year.

    The physical Lance tables remain chronological 50-year buckets. The
    observation index store maps a single publication year onto the
    physical bucket containing that year.

    This preserves the Tier 2 invariant that a seed from year Y is searched
    only against observations from year Y.
    """
    store = LanceObservationIndexStore(
        lance_root,
        available_years=candidate_years,
        available_scales=scales,
    )

    indexes_by_year = {}

    for year in candidate_years:
        indexes = store.get(
            SearchSpace(
                years=(year, year),
                scale=None,
            )
        )

        missing_scales = set(scales) - set(indexes)

        if missing_scales:
            raise RuntimeError(
                f"Missing Lance scale(s) for publication year "
                f"{year}: {sorted(missing_scales)}"
            )

        indexes_by_year[int(year)] = indexes

    return indexes_by_year


def run_lance_tier2(
    *,
    connection,
    concept_name: str,
    concept: dict,
    indexes_by_year,
    candidate_years: tuple[int, ...],
    scales: tuple[str, ...],
    sqlite_path: str | Path = CORPUS_TIER2_DB_PATH,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    batch_size: int = BATCH_SIZE,
    false_positives: list[str] | None = None,
    clear: bool = False,
) -> Path:
    """
    Run one concept and persist its Tier 2 result.

    Retrieval remains in tier2.analysis. This function only resolves the
    workset, consumes batches, and hands the completed result to the
    established SQLite export layer.
    """
    started = time.perf_counter()

    logger.info(
        "[tier2] resolving concept=%s",
        concept_name,
    )

    resolve_started = time.perf_counter()

    resolved = resolve_concept_positions(
        connection=connection,
        concept_name=concept_name,
        concept=concept,
        false_positives=false_positives,
    )

    logger.info(
        "[tier2] resolved concept=%s in %.3fs",
        concept_name,
        time.perf_counter() - resolve_started,
    )

    seed_ids = [
        event_id
        for year in candidate_years
        for event_id in resolved["by_year"].get(
            year,
            (),
        )
    ]

    logger.info(
        "[tier2] query workset: %d seed events, search years=%s-%s",
        len(seed_ids),
        min(candidate_years) if candidate_years else None,
        max(candidate_years) if candidate_years else None,
    )

    output_events = []

    search_started = time.perf_counter()
    batch_count = 0

    for batch in iter_concept_batches(
        connection=connection,
        indexes_by_year=indexes_by_year,
        seed_event_ids=seed_ids,
        scales=scales,
        top_n=top_n,
        rrf_k=rrf_k,
        oversample=oversample,
        false_positives=resolved["false_positives"],
        batch_size=batch_size,
    ):
        output_events.extend(
            batch["events"]
        )
        batch_count += 1

    search_time = (
        time.perf_counter()
        - search_started
    )

    logger.info(
        "[tier2] search complete: %d batches, %d seed events, %.3fs",
        batch_count,
        len(output_events),
        search_time,
    )

    write_started = time.perf_counter()

    sqlite_path = Path(sqlite_path)

    write_tier2_sqlite(
        db_path=sqlite_path,
        concept_name=concept_name,
        events=output_events,
        clear=clear,
    )

    write_time = (
        time.perf_counter()
        - write_started
    )

    total_time = (
        time.perf_counter()
        - started
    )

    logger.info(
        "[tier2] concept=%s timing: search=%.3fs sqlite=%.3fs total=%.3fs",
        concept_name,
        search_time,
        write_time,
        total_time,
    )

    return sqlite_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tier 2 semantic neighbourhood analysis."
    )

    parser.add_argument(
        "--concept",
        help="Run only this concept. Default: all CONCEPT_SETS entries.",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear Tier 2 SQLite output before the first concept.",
    )

    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=K,
        help=f"Number of neighbours per seed (default: {K}).",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help=f"RRF constant (default: {RRF_K}).",
    )

    parser.add_argument(
        "--oversample",
        type=int,
        default=OVERSAMPLE,
        help=f"ANN oversampling factor (default: {OVERSAMPLE}).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Seed batch size (default: {BATCH_SIZE}).",
    )

    parser.add_argument(
        "--false-positives",
        type=str,
        default=None,
        help="Comma-separated forms to exclude.",
    )

    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Earliest publication year to search.",
    )

    parser.add_argument(
        "--to-year",
        type=int,
        default=None,
        help="Latest publication year to search.",
    )

    parser.add_argument(
        "--scale",
        action="append",
        choices=SCALES,
        dest="scales",
        help="Scale to use; may be specified more than once.",
    )

    parser.add_argument(
        "--lance",
        type=str,
        default=str(LANCE_INDEXES_DIR),
        help="Lance index root.",
    )

    parser.add_argument(
        "--sqlite",
        type=str,
        default=str(CORPUS_TIER2_DB_PATH),
        help="Tier 2 SQLite output path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")

    if args.rrf_k <= 0:
        raise ValueError("--rrf-k must be positive")

    if args.oversample <= 0:
        raise ValueError("--oversample must be positive")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if (
        args.from_year is not None
        and args.to_year is not None
        and args.from_year > args.to_year
    ):
        raise ValueError(
            "--from-year cannot be later than --to-year"
        )

    if args.concept:
        concept_name = args.concept.upper()

        if concept_name not in CONCEPT_SETS:
            raise ValueError(
                f"Unknown concept: {concept_name}"
            )

        concept_names = [concept_name]
    else:
        concept_names = list(CONCEPT_SETS)

    requested_scales = (
        tuple(args.scales)
        if args.scales
        else None
    )

    search_space = SearchSpace(
        years=_year_range(
            args.from_year,
            args.to_year,
        ),
        scale=requested_scales,
    )

    logger.info(
        "[tier2] processing %d concept(s)",
        len(concept_names),
    )

    logger.info(
        "[tier2] SQLite output: %s",
        args.sqlite,
    )

    connection = get_connection()

    try:
        available_years = _available_event_years( connection )

        logger.debug( "[tier2] available event years: %s", available_years, )

        (
            candidate_years,
            scales,
        ) = _resolve_search_scope(
            search_space,
            available_years,
        )

        # logger.info( "[tier2] SearchSpace years=%s scales=%s", candidate_years, scales, )

        if not candidate_years:
            logger.warning( "[tier2] no candidate years; nothing to run" )
            return

        index_started = time.perf_counter()

        indexes_by_year = _build_indexes_by_year(
            lance_root=args.lance,
            candidate_years=candidate_years,
            scales=scales,
        )

        logger.info( "[tier2] prepared %d temporal index sets in %.3fs", len(indexes_by_year), time.perf_counter() - index_started, )

        false_positives = (
            [
                value.strip()
                for value in args.false_positives.split(",")
                if value.strip()
            ]
            if args.false_positives
            else None
        )

        for index, concept_name in enumerate(
            concept_names,
            start=1,
        ):
            logger.info(
                "[tier2] ===== concept %d/%d: %s =====",
                index,
                len(concept_names),
                concept_name,
            )

            # --clear belongs only to the first concept; otherwise each
            # subsequent concept would erase the preceding results.
            clear = (
                args.clear
                and index == 1
            )

            run_lance_tier2(
                connection=connection,
                concept_name=concept_name,
                concept=CONCEPT_SETS[concept_name],
                indexes_by_year=indexes_by_year,
                candidate_years=candidate_years,
                scales=scales,
                sqlite_path=args.sqlite,
                top_n=args.k,
                rrf_k=args.rrf_k,
                oversample=args.oversample,
                batch_size=args.batch_size,
                false_positives=false_positives,
                clear=clear,
            )

        logger.info(
            "[tier2] completed %d concept(s)",
            len(concept_names),
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
