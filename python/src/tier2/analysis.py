"""
Tier 2 retrieval and result assembly.

PostgreSQL is authoritative for Tier 1 event identity and provenance.
Lance is authoritative for embedding geometry.

Tier 2 searches each seed only against observations from the seed's
publication year. Temporal restriction therefore belongs to the Lance
search population, while event metadata comes from PostgreSQL.

Tier 2 does not repair missing vectors and does not maintain a second
observation store. A repaired Tier 1 event becomes visible automatically
when the current Lance tables are opened.

Failure mode:
    A seed event may exist in PostgreSQL without a corresponding vector
    in Lance. The caller must treat that as a Tier 1 integrity failure,
    rather than silently reconstructing the observation here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from lib.corpus_logging import logger
from retrieval.lance_search import multiscale_search
from retrieval.models import SCALES

K = 60
RRF_K = 60
OVERSAMPLE = 5
BATCH_SIZE = 32 # 128

_NO_WPOS = -1


def _normalise_forms(values: Iterable[str]) -> set[str]:
    return {
        str(value).lower()
        for value in values
    }


def _fetch_event_metadata(
    connection,
    event_ids: Iterable[int],
) -> dict[int, dict[str, Any]]:
    """
    Fetch Tier 1 provenance from PostgreSQL in one bounded query.

    The event set is bounded by the current Tier 2 batch; this avoids
    repeated individual provenance lookups.
    """
    ids = [int(event_id) for event_id in event_ids]

    if not ids:
        return {}

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                event_id,
                doc_id,
                token,
                token_idx,
                pub_year,
                local_window_id,
                local_window_token_pos,
                medium_window_id,
                medium_window_token_pos,
                broad_window_id,
                broad_window_token_pos
            FROM events
            WHERE event_id = ANY(%s)
            """,
            (ids,),
        )

        rows = cursor.fetchall()

    metadata = {}

    for row in rows:
        (
            event_id,
            doc_id,
            token,
            token_idx,
            pub_year,
            local_window_id,
            local_window_token_pos,
            medium_window_id,
            medium_window_token_pos,
            broad_window_id,
            broad_window_token_pos,
        ) = row

        metadata[int(event_id)] = {
            "event_id": int(event_id),
            "doc_id": str(doc_id),
            "token": str(token),
            "token_idx": int(token_idx),
            "pub_year": int(pub_year),
            "local_window_id": local_window_id,
            "local_window_token_pos": local_window_token_pos,
            "medium_window_id": medium_window_id,
            "medium_window_token_pos": medium_window_token_pos,
            "broad_window_id": broad_window_id,
            "broad_window_token_pos": broad_window_token_pos,
        }

    missing = set(ids) - set(metadata)

    if missing:
        raise RuntimeError(
            "Tier 2 requested event IDs absent from PostgreSQL: "
            f"{sorted(missing)[:10]}"
        )

    return metadata


def resolve_concept_positions(
    *,
    connection,
    concept_name,
    concept,
    false_positives=None,
):
    """
    Resolve lexical seed events directly from PostgreSQL.

    PostgreSQL is the source of truth for event identity. No observation
    store is involved.
    """
    forms = _normalise_forms(
        concept.get("forms", [])
    )

    false_positives = _normalise_forms(
        false_positives
        if false_positives is not None
        else concept.get("false_positives", [])
    )

    logger.info(
        "[tier2] %s forms: %s",
        concept_name,
        sorted(forms)[:50],
    )

    if not forms:
        return {
            "forms": forms,
            "false_positives": false_positives,
            "event_ids": [],
            "event_ids_set": set(),
            "by_year": {},
        }

    with connection.cursor() as cursor:
        if false_positives:
            cursor.execute(
                """
                SELECT event_id, pub_year
                FROM events
                WHERE lower(token) = ANY(%s)
                  AND lower(token) <> ALL(%s)
                """,
                (
                    list(forms),
                    list(false_positives),
                ),
            )
        else:
            cursor.execute(
                """
                SELECT event_id, pub_year
                FROM events
                WHERE lower(token) = ANY(%s)
                """,
                (list(forms),),
            )

        rows = cursor.fetchall()

    event_ids = [
        int(row[0])
        for row in rows
    ]

    by_year: dict[int, list[int]] = defaultdict(list)

    for event_id, year in rows:
        by_year[int(year)].append(int(event_id))

    logger.info(
        "[tier2] %s: %d seed events",
        concept_name,
        len(event_ids),
    )

    return {
        "forms": forms,
        "false_positives": false_positives,
        "event_ids": event_ids,
        "event_ids_set": set(event_ids),
        "by_year": dict(by_year),
    }


def _window_metadata(
    metadata: dict[str, Any],
    scale: str,
):
    window_id = metadata.get(
        f"{scale}_window_id"
    )
    token_pos = metadata.get(
        f"{scale}_window_token_pos"
    )

    if window_id is not None:
        window_id = int(window_id)

    if token_pos is not None:
        token_pos = int(token_pos)

        if token_pos == _NO_WPOS:
            token_pos = None

    return window_id, token_pos


def _build_batch_events(
    *,
    seed_event_ids,
    neighbours,
    metadata_by_id,
    false_positives,
):
    output = []

    for seed_event_id, seed_neighbours in zip(
        seed_event_ids,
        neighbours,
    ):
        seed_event_id = int(seed_event_id)
        seed_metadata = metadata_by_id[seed_event_id]

        neighbours_out = []

        for item in seed_neighbours:
            neighbour_id = int(
                item["event_id"]
            )

            if neighbour_id == seed_event_id:
                continue

            metadata = metadata_by_id.get(neighbour_id)

            if metadata is None:
                raise RuntimeError(
                    "Lance returned an event absent from PostgreSQL: "
                    f"{neighbour_id}"
                )

            token = str(metadata["token"])

            if token.lower() in false_positives:
                continue

            (
                local_window_id,
                local_window_token_pos,
            ) = _window_metadata(
                metadata,
                "local",
            )

            (
                medium_window_id,
                medium_window_token_pos,
            ) = _window_metadata(
                metadata,
                "medium",
            )

            (
                broad_window_id,
                broad_window_token_pos,
            ) = _window_metadata(
                metadata,
                "broad",
            )

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "token": token,
                    "doc_id": metadata["doc_id"],
                    "pub_year": metadata["pub_year"],
                    "token_idx": metadata["token_idx"],
                    "local_window_id": local_window_id,
                    "local_window_token_pos": local_window_token_pos,
                    "medium_window_id": medium_window_id,
                    "medium_window_token_pos": medium_window_token_pos,
                    "broad_window_id": broad_window_id,
                    "broad_window_token_pos": broad_window_token_pos,
                    "score": item["score"],
                    "score_local": item["score_local"],
                    "score_medium": item["score_medium"],
                    "score_broad": item["score_broad"],
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        (
            local_window_id,
            local_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "local",
        )

        (
            medium_window_id,
            medium_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "medium",
        )

        (
            broad_window_id,
            broad_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "broad",
        )

        output.append(
            {
                "event_id": seed_event_id,
                "token": seed_metadata["token"],
                "doc_id": seed_metadata["doc_id"],
                "pub_year": seed_metadata["pub_year"],
                "token_idx": seed_metadata["token_idx"],
                "local_window_id": local_window_id,
                "local_window_token_pos": local_window_token_pos,
                "medium_window_id": medium_window_id,
                "medium_window_token_pos": medium_window_token_pos,
                "broad_window_id": broad_window_id,
                "broad_window_token_pos": broad_window_token_pos,
                "neighbours": neighbours_out,
            }
        )

    return output


def iter_concept_batches(
    *,
    connection,
    indexes_by_year,
    seed_event_ids,
    scales,
    top_n,
    rrf_k,
    oversample,
    false_positives,
    batch_size,
):
    """
    Yield bounded Tier 2 batches.

    PostgreSQL supplies seed identity and provenance. Lance supplies the
    seed vectors and performs temporally restricted ANN retrieval.

    Each seed is searched only against the Lance index for its publication
    year. Multiscale fusion therefore occurs within a single chronological
    population.

    Failure mode:
        A seed year without a corresponding temporal index is a construction
        error. A seed vector missing from Lance is a Tier 1 integrity error.
    """
    if not seed_event_ids:
        return

    false_positives = _normalise_forms(
        false_positives or []
    )

    for start in range(
        0,
        len(seed_event_ids),
        batch_size,
    ):
        seed_batch = [
            int(event_id)
            for event_id in seed_event_ids[
                start:start + batch_size
            ]
        ]

        metadata_by_id = _fetch_event_metadata(
            connection,
            seed_batch,
        )

        year_groups: dict[int, list[int]] = defaultdict(list)

        for local_index, event_id in enumerate(seed_batch):
            year = int(
                metadata_by_id[event_id]["pub_year"]
            )
            year_groups[year].append(local_index)

        batch_events = [
            None
            for _ in seed_batch
        ]

        for year, local_indices in year_groups.items():
            indexes = indexes_by_year.get(year)

            if indexes is None:
                raise RuntimeError(
                    "No temporal indexes available for publication year "
                    f"{year}"
                )

            year_seed_ids = [
                seed_batch[index]
                for index in local_indices
            ]

            # PostgreSQL establishes event identity, but only Lance owns the
            # corresponding embedding. Missing vectors therefore indicate a
            # broken Tier 1 completeness invariant.
            queries_by_scale = {
                scale: indexes[scale].reconstruct_many(
                    year_seed_ids
                )
                for scale in scales
            }

            neighbours = multiscale_search(
                indexes=indexes,
                queries_by_scale=queries_by_scale,
                scales=scales,
                top_n=top_n,
                rrf_k=rrf_k,
                oversample=oversample,
            )

            referenced_ids = set(year_seed_ids)

            for seed_neighbours in neighbours:
                for item in seed_neighbours:
                    referenced_ids.add(
                        int(item["event_id"])
                    )

            metadata_by_id.update(
                _fetch_event_metadata(
                    connection,
                    referenced_ids - set(metadata_by_id),
                )
            )

            events = _build_batch_events(
                seed_event_ids=year_seed_ids,
                neighbours=neighbours,
                metadata_by_id=metadata_by_id,
                false_positives=false_positives,
            )

            for local_index, event in zip(
                local_indices,
                events,
            ):
                batch_events[local_index] = event

        yield {
            "type": "batch",
            "events": batch_events,
        }
