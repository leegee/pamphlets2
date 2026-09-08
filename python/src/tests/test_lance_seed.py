from __future__ import annotations

import numpy as np

from lib.corpus_config import (
    EVENTSTORE_T1_PATH,
    LANCE_INDEXES_DIR,
)
from retrieval.lance_observation_index import LanceObservationIndex
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SearchSpace
from retrieval.models import SCALES
from tier1.observation_store_api import (
    open_observation_lookup,
)


SEED_EVENT_ID = 8837468744104859267
K = 10


def get_lookup_vector(
    lookup,
    event_id: int,
    scale: str,
) -> np.ndarray:
    vectors = lookup._scale_for_ids(
        np.asarray([event_id], dtype=np.uint64),
        scale,
    )

    if len(vectors) != 1:
        raise RuntimeError(
            f"Expected one vector for event_id={event_id}, "
            f"got {len(vectors)}"
        )

    vector = np.asarray(
        vectors[0],
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError(
            f"Invalid vector for event_id={event_id}, scale={scale}"
        )

    return vector / norm


def run_bypass(
    index: LanceObservationIndex,
    query_vector: np.ndarray,
    k: int,
):
    request = (
        index._table
        .search(
            query_vector,
            vector_column_name="vector",
        )
        .limit(k)
        .bypass_vector_index()
    )

    return request.to_list()


def print_results(
    title: str,
    rows,
) -> None:
    print()
    print(title)
    print("-" * 70)

    for rank, row in enumerate(rows, start=1):
        event_id = int(row["event_id"])

        distance = row.get("_distance")

        if distance is None:
            distance = row.get("distance")

        print(
            f"{rank:2d}. "
            f"id={event_id} "
            f"distance={float(distance):.8f}"
        )


def compare(
    *,
    lookup,
    index: LanceObservationIndex,
    event_id: int,
    scale: str,
) -> None:
    query_vector = get_lookup_vector(
        lookup,
        event_id,
        scale,
    )

    print()
    print("=" * 70)
    print(f"scale={scale}")
    print(f"seed={event_id}")
    print("=" * 70)

    # IVF-PQ is the production ANN path.
    ann_result = index.search(
        query_vector,
        k=K,
    )

    print()
    print("IVF-PQ ANN")
    print("-" * 70)

    for rank, (result_id, distance) in enumerate(
        zip(
            ann_result.event_ids,
            ann_result.distances,
        ),
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"id={int(result_id)} "
            f"distance={float(distance):.8f}"
        )

    # This bypasses the vector index while querying the same Lance table.
    bypass_rows = run_bypass(
        index,
        query_vector,
        K,
    )

    print_results(
        "Lance bypass_vector_index()",
        bypass_rows,
    )

    ann_ids = [
        int(event_id)
        for event_id in ann_result.event_ids
    ]

    bypass_ids = [
        int(row["event_id"])
        for row in bypass_rows
    ]

    ann_set = set(ann_ids)
    bypass_set = set(bypass_ids)

    intersection = ann_set & bypass_set

    print()
    print("Comparison")
    print("-" * 70)
    print(f"IVF-PQ IDs:  {ann_ids}")
    print(f"Bypass IDs:  {bypass_ids}")
    print(
        f"Overlap:     {len(intersection)}/{K}"
    )

    if ann_ids and bypass_ids:
        print(
            f"ANN rank-1:     {ann_ids[0]}"
        )
        print(
            f"Bypass rank-1:  {bypass_ids[0]}"
        )

        if ann_ids[0] == bypass_ids[0]:
            print("RESULT: rank-1 agrees")
        else:
            print("RESULT: rank-1 differs")


def main() -> None:
    lookup = open_observation_lookup(
        EVENTSTORE_T1_PATH
    )

    store = LanceObservationIndexStore(
        LANCE_INDEXES_DIR,
        available_years=lookup.available_years,
        available_scales=SCALES,
        dimensions=768,
        nprobes=500,
        model="macberth",
    )

    indexes = store.get(
        SearchSpace(
            years=None,
            scale=tuple(SCALES),
        )
    )

    for scale in (
        "local",
        "medium",
        "broad",
    ):
        compare(
            lookup=lookup,
            index=indexes[scale],
            event_id=SEED_EVENT_ID,
            scale=scale,
        )


if __name__ == "__main__":
    main()