from __future__ import annotations

import numpy as np
from collections.abc import Iterator
from typing import Literal

from retrieval.models import Float32Array, SearchResult, SearchSpace
from retrieval.observation_index_store import ObservationIndexStore
from retrieval.parquet_context import ObservationContext, ParquetContext
from retrieval.retriever import ObservationRetriever


class IndexedObservationRetriever(ObservationRetriever):
    """Resolve searches in selected observation spaces into corpus context."""

    def __init__(
        self,
        index_store: ObservationIndexStore,
        context: ParquetContext,
    ) -> None:
        self._index_store = index_store
        self._context = context


    def diachronic_search(
        self,
        query: Float32Array,
        *,
        space: SearchSpace,
        k: int,
        direction: Literal["forward", "backward"] = "forward",
    ) -> Iterator[tuple[tuple[int, int], list[ObservationContext]]]:
        """Search the requested chronology one physical bucket at a time.

        Bucket boundaries are owned by the index store; callers specify only
        the logical chronological search space and traversal direction.

        A single query vector is reused across all requested scales. This is
        appropriate for phrase queries, where the query has one representation
        while the observation index may expose multiple retrieval scales.

        Results from all requested scales are merged within each bucket and
        ranked by distance. Observations from different buckets are never
        combined into one global ranking.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        scales = space.resolve_scales(
            set(self._index_store.available_scales)
        )

        queries_by_scale = {
            scale: query
            for scale in scales
        }

        for bucket, results_by_scale in self._index_store.diachronic_search(
            queries_by_scale,
            space,
            k=k,
            direction=direction,
        ):
            results: list[ObservationContext] = []

            for scale in scales:
                results.extend(
                    self._context.get_many(
                        results_by_scale[scale],
                    )
                )

            best_by_event_id: dict[int, ObservationContext] = {}

            for result in results:
                existing = best_by_event_id.get(result.event_id)

                if existing is None or result.distance < existing.distance:
                    best_by_event_id[result.event_id] = result

            results = list(best_by_event_id.values())

            results.sort(
                key=lambda result: result.distance,
            )

            if len(results) > k:
                results = results[:k]

            yield bucket, results


    def search(
        self,
        query: Float32Array,
        *,
        space: SearchSpace,
        k: int,
    ) -> list[ObservationContext]:
        if k <= 0:
            raise ValueError("k must be positive")

        indexes = self._index_store.get(space)

        if not indexes:
            return []

        results = [
            index.search(
                query,
                k=k,
            )
            for index in indexes.values()
        ]

        event_ids = np.concatenate(
            [result.event_ids for result in results],
        )
        distances = np.concatenate(
            [result.distances for result in results],
        )

        if len(event_ids) > k:
            selected = np.argpartition(
                distances,
                k - 1,
            )[:k]

            selected = selected[
                np.argsort(distances[selected])
            ]
        else:
            selected = np.argsort(distances)

        merged = SearchResult(
            event_ids=event_ids[selected],
            distances=distances[selected],
        )

        return self._context.get_many(
            merged,
        )


    def batch_search(
        self,
        queries: Float32Array,
        *,
        space: SearchSpace,
        k: int,
        oversample: int,
    ) -> list[list[ObservationContext]]:
        if k <= 0:
            raise ValueError("k must be positive")

        indexes = self._index_store.get(space)

        if not indexes:
            return [
                []
                for _ in range(len(queries))
            ]

        batch_results = [
            index.batch_search(
                queries,
                k=k,
                oversample=oversample,
            )
            for index in indexes.values()
        ]

        merged_results: list[SearchResult] = []

        for query_idx in range(len(queries)):
            event_ids = np.concatenate(
                [
                    results[query_idx].event_ids
                    for results in batch_results
                ],
            )

            distances = np.concatenate(
                [
                    results[query_idx].distances
                    for results in batch_results
                ],
            )

            if len(event_ids) > k:
                selected = np.argpartition(
                    distances,
                    k - 1,
                )[:k]

                selected = selected[
                    np.argsort(distances[selected])
                ]
            else:
                selected = np.argsort(distances)

            merged_results.append(
                SearchResult(
                    event_ids=event_ids[selected],
                    distances=distances[selected],
                )
            )

        return [
            self._context.get_many(result)
            for result in merged_results
        ]
