# tier1/observation_workset.py
from __future__ import annotations

import time
import argparse
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.corpus_config import (
    CONCEPT_SETS,
    EVENTSTORE_T1_PATH,
    LANCE_INDEXES_DIR,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.models import SearchSpace
from tier1.observation_store_api import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    SCALES,
    open_observation_lookup,
)

# These are deliberately the same retrieval parameters used by the current
# multiscale Lance search. They determine the candidate population, not a
# semantic classification threshold.
TOP_K = 60
RRF_K = 60
OVERSAMPLE = 5
NPROBES = 150

Occurrence = tuple[str, str, int]


def normalise_form(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().lower()


@dataclass(frozen=True)
class ConceptSeeds:
    concept: str
    forms: frozenset[str]
    false_positives: frozenset[str]


class ObservationWorkset:
    """Construct a Tier 1 population from lexical seeds and their neighbours."""

    def __init__(
        self,
        conn,
        *,
        store_path: str | Path = EVENTSTORE_T1_PATH,
        lance_root: str | Path = LANCE_INDEXES_DIR,
        top_k: int = TOP_K,
        rrf_k: int = RRF_K,
        oversample: int = OVERSAMPLE,
        nprobes: int = NPROBES,
    ):
        self.conn = conn
        self.store_path = Path(store_path)
        self.lance_root = Path(lance_root)
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.oversample = oversample
        self.nprobes = nprobes

    def configured_concepts(self) -> tuple[ConceptSeeds, ...]:
        concepts = []

        for concept, rule in CONCEPT_SETS.items():
            concepts.append(
                ConceptSeeds(
                    concept=concept,
                    forms=frozenset(
                        normalise_form(form)
                        for form in rule.get("forms", set())
                    ),
                    false_positives=frozenset(
                        normalise_form(form)
                        for form in rule.get("false_positives", set())
                    ),
                )
            )

        return tuple(concepts)

    def seeds(
        self,
        *,
        concepts: list[str] | None = None,
    ) -> dict[str, set[Occurrence]]:
        requested = (
            {concept.upper() for concept in concepts}
            if concepts is not None
            else None
        )

        output: dict[str, set[Occurrence]] = {}

        for rule in self.configured_concepts():
            if requested is not None and rule.concept.upper() not in requested:
                continue

            if not rule.forms:
                output[rule.concept] = set()
                logger.info(
                    "[tier1] concept=%s has no configured seed forms",
                    rule.concept,
                )
                continue

            matches = self._find_exact_seeds(
                rule.forms,
                rule.false_positives,
            )

            output[rule.concept] = matches

            logger.info(
                "[tier1] concept=%s seeds=%d forms=%s false_positives=%d",
                rule.concept,
                len(matches),
                sorted(rule.forms),
                len(rule.false_positives),
            )

        return output

    def _find_exact_seeds(
        self,
        forms: frozenset[str],
        false_positives: frozenset[str],
    ) -> set[Occurrence]:
        """Return corpus coordinates matching configured lexical forms."""

        if not forms:
            return set()

        rows = self.conn.execute(
            """
            SELECT
                corpus,
                doc_id,
                token_idx,
                token
            FROM pamphlet_tokens
            WHERE lower(token) = ANY(%s)
            """,
            (list(forms),),
        ).fetchall()

        occurrences: set[Occurrence] = set()

        for corpus, doc_id, token_idx, token in rows:
            normalised = normalise_form(token)

            if normalised in false_positives:
                continue

            if normalised not in forms:
                continue

            occurrences.add(
                (
                    str(corpus),
                    str(doc_id),
                    int(token_idx),
                )
            )

        return occurrences

    def expand(
        self,
        seed_occurrences: set[Occurrence],
        *,
        lookup=None,
        lance_store=None,
    ) -> tuple[set[Occurrence], dict[int, list[dict]]]:
        """
        Expand lexical seed occurrences through the existing Tier 1 geometry.

        The returned workset contains the seeds themselves plus the unique
        semantic neighbours returned by multiscale Lance retrieval.

        Neighbour metadata are retained separately so the caller can inspect
        what was selected without making provenance part of the embedding
        input contract.
        """

        if not seed_occurrences:
            return set(), {}

        expand_started = time.perf_counter()

        lookup_started = time.perf_counter()
        lookup = lookup or open_observation_lookup(self.store_path)
        logger.info( "[tier1] workset: opened observation lookup in %.3fs", time.perf_counter() - lookup_started, )

        event_started = time.perf_counter()
        occurrence_to_events = lookup.find_event_ids_by_positions(
            sorted(seed_occurrences)
        )

        seed_event_ids = sorted(
            {
                int(event_id)
                for event_ids in occurrence_to_events.values()
                for event_id in event_ids
            }
        )

        logger.info(
            "[tier1] workset: resolved %d occurrences -> %d events in %.3fs",
            len(seed_occurrences),
            len(seed_event_ids),
            time.perf_counter() - event_started,
        )

        if not seed_event_ids:
            raise RuntimeError( "None of the configured seed occurrences are present in the existing Tier 1 observation store" )

        available_years = {
            int(year)
            for year in lookup.available_years
        }

        if not available_years:
            raise RuntimeError( "Existing Tier 1 observation store contains no publication years" )

        lance_started = time.perf_counter()

        lance_store = lance_store or LanceObservationIndexStore(
            self.lance_root,
            available_years=available_years,
            nprobes=self.nprobes,
        )

        logger.info( "[tier1] workset: opened Lance store in %.3fs", time.perf_counter() - lance_started, )

        index_started = time.perf_counter()

        search_space = SearchSpace(
            years=None,
            scale=SCALES,
        )

        indexes = lance_store.get(search_space)

        logger.info( "[tier1] workset: resolved search indexes in %.3fs", time.perf_counter() - index_started, )

        query_started = time.perf_counter()
        scale_vectors: dict[str, np.ndarray] = {}

        for scale in SCALES:
            scale_started = time.perf_counter()

            vectors = indexes[scale].reconstruct_many(seed_event_ids)

            if vectors.shape != (len(seed_event_ids), 768):
                raise RuntimeError(
                    f"Lance reconstruction for scale={scale!r} returned "
                    f"shape {vectors.shape}; expected "
                    f"({len(seed_event_ids)}, 768)"
                )

            scale_vectors[scale] = vectors

            logger.info(
                "[tier1] workset: reconstructed %d %s query embeddings "
                "from Lance in %.3fs",
                len(seed_event_ids),
                scale,
                time.perf_counter() - scale_started,
            )

        queries = np.zeros(
            (len(seed_event_ids), 768),
            dtype=np.float32,
        )

        for scale, weight in zip(
            SCALES,
            DEFAULT_ENSEMBLE_WEIGHTS,
        ):
            queries += ( np.float32(weight) * scale_vectors[scale] )

        logger.info(
            "[tier1] workset: assembled %d ensemble query embeddings "
            "from Lance in %.3fs",
            len(queries),
            time.perf_counter() - query_started,
        )

        if len(queries) != len(seed_event_ids):
            raise RuntimeError(
                "Lance reconstruction returned a different number of "
                "query vectors than seed event IDs"
            )

        neighbours_by_seed: dict[int, list[dict]] = {}
        neighbour_event_ids: set[int] = set()

        search_k = self.top_k * self.oversample
        batch_size = 32
        total_batches = (
            len(seed_event_ids) + batch_size - 1
        ) // batch_size

        retrieval_started = time.perf_counter()

        for batch_number, start in enumerate(
            range(0, len(seed_event_ids), batch_size),
            start=1,
        ):
            batch_started = time.perf_counter()

            batch_ids = seed_event_ids[start:start + batch_size]
            batch_queries = queries[start:start + batch_size]

            per_seed: list[dict[int, dict]] = [
                {}
                for _ in batch_ids
            ]

            scale_timings: dict[str, float] = {}

            for scale in SCALES:
                scale_started = time.perf_counter()

                index = indexes[scale]

                results = index.batch_search(
                    batch_queries,
                    k=search_k + 1,
                )

                scale_timings[scale] = ( time.perf_counter() - scale_started )

                for row_idx, seed_event_id in enumerate(batch_ids):
                    fused = per_seed[row_idx]

                    for rank, (
                        event_id,
                        distance,
                    ) in enumerate(
                        zip(
                            results.event_ids[row_idx],
                            results.distances[row_idx],
                        ),
                        start=1,
                    ):
                        event_id = int(event_id)

                        if event_id == seed_event_id:
                            continue

                        item = fused.setdefault(
                            event_id,
                            {
                                "event_id": event_id,
                                "rrf_score": 0.0,
                                "score_local": None,
                                "score_medium": None,
                                "score_broad": None,
                            },
                        )

                        item["rrf_score"] += (
                            1.0 / (self.rrf_k + rank)
                        )
                        item[f"score_{scale}"] = float(distance)

            for seed_event_id, fused in zip(batch_ids, per_seed):
                ranked = sorted(
                    fused.values(),
                    key=lambda item: item["rrf_score"],
                    reverse=True,
                )[:self.top_k]

                neighbours_by_seed[seed_event_id] = ranked

                for item in ranked:
                    neighbour_event_ids.add(
                        int(item["event_id"])
                    )

            batch_elapsed = time.perf_counter() - batch_started
            retrieval_elapsed = time.perf_counter() - retrieval_started

            completed = min(
                start + len(batch_ids),
                len(seed_event_ids),
            )

            rate = completed / retrieval_elapsed
            remaining = len(seed_event_ids) - completed
            eta = remaining / rate if rate > 0 else 0.0

            logger.info(
                "[tier1] batch %d/%d: seeds=%d elapsed=%.3fs "
                "local=%.3fs medium=%.3fs broad=%.3fs "
                "rate=%.2f seeds/s ETA=%.1fs",
                batch_number,
                total_batches,
                len(batch_ids),
                batch_elapsed,
                scale_timings.get("local", 0.0),
                scale_timings.get("medium", 0.0),
                scale_timings.get("broad", 0.0),
                rate,
                eta,
            )

        logger.info(
            "[tier1] semantic expansion complete: "
            "%d seeds in %.3fs",
            len(seed_event_ids),
            time.perf_counter() - retrieval_started,
        )

        metadata_started = time.perf_counter()

        neighbour_occurrences: set[Occurrence] = set()

        for event_id in neighbour_event_ids:
            metadata = lookup.get_event_metadata(event_id)

            neighbour_occurrences.add(
                (
                    str(metadata["corpus"]),
                    str(metadata["doc_id"]),
                    int(metadata["token_idx"]),
                )
            )

        logger.info(
            "[tier1] workset: resolved %d neighbour events -> "
            "%d occurrences in %.3fs",
            len(neighbour_event_ids),
            len(neighbour_occurrences),
            time.perf_counter() - metadata_started,
        )

        workset = set(seed_occurrences)
        workset.update(neighbour_occurrences)

        logger.info(
            "[tier1] workset expansion total time %.3fs",
            time.perf_counter() - expand_started,
        )

        return workset, neighbours_by_seed


    def build(
        self,
        *,
        concepts: list[str] | None = None,
    ) -> tuple[dict[str, set[Occurrence]], dict[str, dict]]:
        """
        Build concept-specific Tier 1 worksets.

        Each concept is expanded independently so that later inspection can
        distinguish which lexical seed generated an observation.
        """

        seed_sets = self.seeds(concepts=concepts)
        output: dict[str, set[Occurrence]] = {}
        diagnostics: dict[str, dict] = {}

        lookup = open_observation_lookup( self.store_path )

        available_years = {
            int(year)
            for year in lookup.available_years
        }

        lance_store = LanceObservationIndexStore(
            self.lance_root,
            available_years=available_years,
            nprobes=self.nprobes,
        )

        for concept, seed_occurrences in seed_sets.items():
            if not seed_occurrences:
                output[concept] = set()
                diagnostics[concept] = {
                    "seeds": 0,
                    "neighbour_events": 0,
                    "workset": 0,
                    "neighbours_by_seed": {},
                }
                continue

            workset, neighbours_by_seed = self.expand(
                seed_occurrences,
                lookup=lookup,
                lance_store=lance_store,
            )

            output[concept] = workset

            diagnostics[concept] = {
                "seeds": len(seed_occurrences),
                "neighbour_events": sum(
                    len(values)
                    for values in neighbours_by_seed.values()
                ),
                "unique_neighbour_events": len(
                    {
                        int(event["event_id"])
                        for values in neighbours_by_seed.values()
                        for event in values
                    }
                ),
                "workset": len(workset),
                "neighbours_by_seed": neighbours_by_seed,
            }

            logger.info(
                "[tier1] concept=%s seeds=%d unique_neighbours=%d workset=%d",
                concept,
                diagnostics[concept]["seeds"],
                diagnostics[concept]["unique_neighbour_events"],
                diagnostics[concept]["workset"],
            )

        return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept",
        action="append",
        dest="concepts",
        default=None,
        help="Concept to process; may be supplied more than once.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=OVERSAMPLE,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    conn = get_connection()

    try:
        workset_builder = ObservationWorkset(
            conn,
            top_k=args.k,
            rrf_k=args.rrf_k,
            oversample=args.oversample,
        )

        worksets, diagnostics = workset_builder.build(
            concepts=args.concepts,
        )

        lookup = open_observation_lookup( EVENTSTORE_T1_PATH )

        for concept, workset in worksets.items():
            stats = diagnostics[concept]

            logger.info("")
            logger.info("=" * 70)
            logger.info("[tier1] CONCEPT %s", concept)
            logger.info("=" * 70)
            logger.info( "[tier1] seeds:                 %d", stats["seeds"] )
            logger.info( "[tier1] unique neighbours:     %d", stats["unique_neighbour_events"] )
            logger.info( "[tier1] final workset:          %d", stats["workset"] )

            if not stats["neighbours_by_seed"]:
                continue

            sample_seed_id = next( iter(stats["neighbours_by_seed"]) )

            seed_metadata = lookup.get_event_metadata( sample_seed_id )

            logger.info("")
            logger.info( "[tier1] SAMPLE SEED: %r (%s)", seed_metadata["token"], seed_metadata["doc_id"], )

            for rank, item in enumerate(
                stats["neighbours_by_seed"][sample_seed_id][
                    :args.sample
                ],
                start=1,
            ):
                metadata = lookup.get_event_metadata( int(item["event_id"]) )

                logger.info(
                    "%3d. rrf=%.6f distance_medium=%s token=%r doc=%s",
                    rank,
                    item["rrf_score"],
                    (
                        f"{item['score_medium']:.6f}"
                        if item["score_medium"] is not None
                        else "—"
                    ),
                    metadata["token"],
                    metadata["doc_id"],
                )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
