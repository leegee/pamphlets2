"""
tier2/sqlite.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from lib.corpus_logging import logger
from lib.corpus_db import analysis_db_connection


_SCHEMA_INIT = """
CREATE TABLE IF NOT EXISTS concepts (
    concept  TEXT PRIMARY KEY,
    n_events INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id         INTEGER PRIMARY KEY,
    concept          TEXT    NOT NULL,
    vector_id        INTEGER,
    token            TEXT,
    doc_id           TEXT,
    pub_year         INTEGER,
    token_idx        INTEGER,
    window_id        INTEGER,
    window_token_pos INTEGER,

    nx               REAL,
    ny               REAL,
    gnx              REAL,
    gny              REAL,
    cluster_id       INTEGER,
    cluster_label    TEXT,

    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE TABLE IF NOT EXISTS neighbours (
    event_id             INTEGER NOT NULL,
    neighbour_event_id   INTEGER NOT NULL,
    vector_id            INTEGER,
    token                TEXT,
    doc_id               TEXT,
    pub_year             INTEGER,
    token_idx             INTEGER,
    window_id             INTEGER,
    window_token_pos     INTEGER,
    score                 REAL,
    score_local            REAL,
    score_medium           REAL,
    score_broad            REAL,

    PRIMARY KEY (event_id, neighbour_event_id),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE TABLE IF NOT EXISTS concept_field_events (
    concept  TEXT    NOT NULL,
    event_id INTEGER NOT NULL,
    role     TEXT    NOT NULL,

    PRIMARY KEY (concept, event_id),
    FOREIGN KEY (concept) REFERENCES concepts(concept),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE TABLE IF NOT EXISTS concept_aggregate (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    concept       TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    rank          INTEGER NOT NULL,
    value         TEXT,
    window_doc_id TEXT,
    window_id     INTEGER,
    count         INTEGER NOT NULL,

    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE TABLE IF NOT EXISTS concept_cluster_info (
    concept          TEXT    NOT NULL,
    cluster_id       INTEGER NOT NULL,
    cluster_label    TEXT,
    centroid_nx      REAL,
    centroid_ny      REAL,
    centroid_gnx     REAL,
    centroid_gny     REAL,
    centroid_vector  BLOB,
    point_count      INTEGER NOT NULL,
    description      TEXT,

    PRIMARY KEY (concept, cluster_id),
    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE INDEX IF NOT EXISTS idx_events_concept
    ON events(concept);

CREATE INDEX IF NOT EXISTS idx_events_token
    ON events(token);

CREATE INDEX IF NOT EXISTS idx_events_event_id
    ON events(event_id);

CREATE INDEX IF NOT EXISTS idx_events_doc_id
    ON events(doc_id);

CREATE INDEX IF NOT EXISTS idx_events_concept_year
    ON events(concept, pub_year);

CREATE INDEX IF NOT EXISTS idx_neighbours_event_id
    ON neighbours(event_id);

CREATE INDEX IF NOT EXISTS idx_neighbours_token
    ON neighbours(token);

CREATE INDEX IF NOT EXISTS idx_field_events_concept
    ON concept_field_events(concept);

CREATE INDEX IF NOT EXISTS idx_field_events_event
    ON concept_field_events(event_id);

CREATE INDEX IF NOT EXISTS idx_aggregate_concept
    ON concept_aggregate(concept, kind);
"""

_SCHEMA_CLEAR = """
DROP TABLE IF EXISTS concept_cluster_info;
DROP TABLE IF EXISTS concept_aggregate;
DROP TABLE IF EXISTS concept_field_events;
DROP TABLE IF EXISTS neighbours;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS concepts;
"""


_DELETE_CONCEPT = (
    "DELETE FROM concept_cluster_info WHERE concept = ?",
    "DELETE FROM concept_aggregate WHERE concept = ?",
    """
    DELETE FROM neighbours
    WHERE event_id IN (
        SELECT event_id FROM events WHERE concept = ?
    )
    """,
    "DELETE FROM concept_field_events WHERE concept = ?",
    "DELETE FROM events WHERE concept = ?",
    "DELETE FROM concepts WHERE concept = ?",
)

def _maybe_float(value):
    return None if value is None else float(value)


def _aggregate_rows(
    concept_name: str,
    events: list[dict],
):
    """
    Yield concept_aggregate rows for the retrieved neighbourhood.

    Token and document rankings are weighted by each neighbour's RRF-fused
    score, not just by how many seed events happened to retrieve it. RRF
    score is derived from rank position within each scale's L2 search
    results, which makes it comparable across neighbours even though raw
    L2 distance is not directly comparable across queries or scales
    (distance scale depends on embedding dimensionality and vector norm,
    neither of which this aggregation should assume). This keeps
    high-frequency, weakly-related tokens (short function words, numbers,
    etc. that surface as "hub" points across many unrelated searches) from
    dominating the aggregate ahead of genuinely close matches.

    Windows retain retrieval-count semantics because each local window
    represents a concrete retrieved relationship.

    Failure mode:
        A neighbour without a local_window_id cannot contribute to
        top_windows, but remains eligible for token and document aggregates.
    """
    token_seed_weight: dict[str, dict[int, float]] = {}
    doc_seed_weight: dict[str, dict[int, float]] = {}
    window_counts: Counter[tuple[str, int]] = Counter()

    for event in events:
        event_id = int(event["event_id"])

        for neighbour in event.get("neighbours", []):
            weight = float(neighbour.get("score", 0.0))

            token = neighbour.get("token")

            if token is not None:
                token = str(token)
                per_event = token_seed_weight.setdefault(token, {})
                # A seed event may retrieve the same token more than once;
                # keep its single best (highest-weight) contribution.
                per_event[event_id] = max(
                    per_event.get(event_id, 0.0),
                    weight,
                )

            doc_id = neighbour.get("doc_id")

            if doc_id is not None:
                doc_id = str(doc_id)
                per_event = doc_seed_weight.setdefault(doc_id, {})
                per_event[event_id] = max(
                    per_event.get(event_id, 0.0),
                    weight,
                )

            window_id = neighbour.get("local_window_id")

            if doc_id is not None and window_id is not None:
                window_counts[(doc_id, int(window_id))] += 1

    token_ranked = sorted(
        token_seed_weight.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    for rank, (token, per_event) in enumerate(token_ranked):
        yield (
            concept_name,
            "token",
            rank,
            token,
            None,
            None,
            len(per_event),
        )

    doc_ranked = sorted(
        doc_seed_weight.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    for rank, (doc_id, per_event) in enumerate(doc_ranked):
        yield (
            concept_name,
            "doc",
            rank,
            doc_id,
            None,
            None,
            len(per_event),
        )

    for rank, ((doc_id, window_id), count) in enumerate(
        window_counts.most_common()
    ):
        yield (
            concept_name,
            "window",
            rank,
            None,
            doc_id,
            window_id,
            count,
        )



def write_tier2_sqlite(
    *,
    db_path: str | Path,
    concept_name: str,
    events: list[dict],
    clear: bool = False,
):
    """
    Write one concept's Tier 2 results to the established SQLite schema.

    Existing rows for this concept are removed before replacement so repeated
    runs remain idempotent without disturbing other concepts.

    Failure mode:
        neighbour membership belongs to the seed event. The same neighbour
        may therefore legitimately appear under many rows in neighbours.

        Aggregates are derived from those same relationships so their meaning
        cannot diverge from the persisted neighbour data.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir( parents=True, exist_ok=True, )

    logger.info( "[tier2] writing sqlite -> %s", db_path, )

    con = analysis_db_connection(db_path)

    try:
        con.execute("PRAGMA foreign_keys = ON")

        if clear:
            logger.info("[tier2] clearing sqlite database")
            con.executescript(_SCHEMA_CLEAR)

        con.executescript(_SCHEMA_INIT)

        con.execute("BEGIN")

        for index, statement in enumerate(_DELETE_CONCEPT):
            logger.info( "[tier2] deleting concept=%s phase=%d", concept_name, index, )
            con.execute( statement, (concept_name,), )

        con.execute(
            """
            INSERT INTO concepts (
                concept,
                n_events
            )
            VALUES (?, ?)
            """,
            (
                concept_name,
                len(events),
            ),
        )

        event_rows = []
        field_event_rows = []
        neighbour_rows = []

        for event in events:
            event_id = int(event["event_id"])

            event_rows.append(
                (
                    event_id,
                    concept_name,
                    None,
                    event["token"],
                    event["doc_id"],
                    int(event["pub_year"]),
                    int(event["token_idx"]),
                    event["local_window_id"],
                    event["local_window_token_pos"],
                )
            )

            field_event_rows.append(
                (
                    concept_name,
                    event_id,
                    'seed',
                )
            )

            for neighbour in event.get("neighbours", []):
                neighbour_rows.append(
                    (
                        event_id,
                        int(neighbour["event_id"]),
                        None,
                        neighbour["token"],
                        neighbour["doc_id"],
                        int(neighbour["pub_year"]),
                        int(neighbour["token_idx"]),
                        neighbour["local_window_id"],
                        neighbour["local_window_token_pos"],
                        float(neighbour["score"]),
                        _maybe_float(neighbour.get("score_local")),
                        _maybe_float(neighbour.get("score_medium")),
                        _maybe_float(neighbour.get("score_broad")),
                    )
                )
        con.executemany(
            """
            INSERT INTO events (
                event_id,
                concept,
                vector_id,
                token,
                doc_id,
                pub_year,
                token_idx,
                window_id,
                window_token_pos
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )

        con.executemany(
            """
            INSERT INTO concept_field_events (
                concept,
                event_id,
                role
            )
            VALUES (?, ?, ?)
            """,
            field_event_rows,
        )

        con.executemany(
            """
            INSERT INTO neighbours (
                event_id,
                neighbour_event_id,
                vector_id,
                token,
                doc_id,
                pub_year,
                token_idx,
                window_id,
                window_token_pos,
                score,
                score_local,
                score_medium,
                score_broad
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            neighbour_rows,
        )

        aggregate_rows = list(
            _aggregate_rows(
                concept_name,
                events,
            )
        )

        if aggregate_rows:
            con.executemany(
                """
                INSERT INTO concept_aggregate (
                    concept,
                    kind,
                    rank,
                    value,
                    window_doc_id,
                    window_id,
                    count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                aggregate_rows,
            )

        con.commit()

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()

    logger.info(
        "[tier2] sqlite write complete: "
        "concept=%s events=%d neighbours=%d",
        concept_name,
        len(events),
        len(neighbour_rows),
    )
