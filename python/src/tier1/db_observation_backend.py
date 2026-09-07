# tier1/db_observation_backend.py

from typing import Sequence

EVENT_COLUMNS = (
    "event_id",
    "corpus",
    "doc_id",
    "token",
    "token_idx",
    "pub_year",
    "local_window_id",
    "local_window_token_pos",
    "medium_window_id",
    "medium_window_token_pos",
    "broad_window_id",
    "broad_window_token_pos",
)


def create_events_table(conn: Connection) -> None:
    """
    Create the durable observation identity and metadata table.

    event_id is the stable identity shared by PostgreSQL and Lance.
    Vector data is deliberately not stored here.
    """
    logger.info("[corpus_db] Creating events table")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                CREATE SEQUENCE IF NOT EXISTS event_id_seq;

                CREATE TABLE IF NOT EXISTS events (
                    event_id BIGINT PRIMARY KEY
                        DEFAULT nextval('event_id_seq'),

                    corpus TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,
                    pub_year INTEGER,

                    local_window_id BIGINT,
                    local_window_token_pos INTEGER,

                    medium_window_id BIGINT,
                    medium_window_token_pos INTEGER,

                    broad_window_id BIGINT,
                    broad_window_token_pos INTEGER,

                    CONSTRAINT events_position_unique
                        UNIQUE (corpus, doc_id, token_idx),

                    CONSTRAINT events_document_fk
                        FOREIGN KEY (doc_id)
                        REFERENCES documents(doc_id)
                        ON DELETE CASCADE
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_corpus_doc_token
                ON events(corpus, doc_id, token_idx);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_doc_token
                ON events(doc_id, token_idx);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_pub_year
                ON events(pub_year);
            """)

    logger.info("[corpus_db] Events table created")


def drop_events_table(conn: Connection) -> None:
    """
    Drop the event table and its ID sequence.

    This is separate from init_db() because existing Tier 1 observations
    may need to be backfilled without rebuilding the corpus database.
    """
    logger.info("[corpus_db] Dropping events table")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS events CASCADE;")
            cur.execute("DROP SEQUENCE IF EXISTS event_id_seq;")

    logger.info("[corpus_db] Events table dropped")


def allocate_event_ids(
    conn: Connection,
    count: int,
) -> list[int]:
    """
    Reserve a contiguous block of event IDs.

    Sequence gaps are acceptable: event IDs are stable identities, not
    row numbers, and allocation may occur before a corresponding Lance
    write succeeds.
    """
    if count < 0:
        raise ValueError("count must be non-negative")

    if count == 0:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT nextval('event_id_seq')
            FROM generate_series(1, %s);
            """,
            (count,),
        )
        return [int(row[0]) for row in cur.fetchall()]


def insert_events(
    conn: Connection,
    *,
    event_id: Sequence[int],
    corpus: Sequence[str],
    doc_id: Sequence[str],
    token: Sequence[str],
    token_idx: Sequence[int],
    pub_year: Sequence[int | None],
    local_window_id: Sequence[int | None] | None = None,
    local_window_token_pos: Sequence[int | None] | None = None,
    medium_window_id: Sequence[int | None] | None = None,
    medium_window_token_pos: Sequence[int | None] | None = None,
    broad_window_id: Sequence[int | None] | None = None,
    broad_window_token_pos: Sequence[int | None] | None = None,
) -> None:
    """
    Insert a batch of event identities and metadata.

    Vector data is intentionally absent: vectors are written directly to
    Lance using the same event IDs.
    """
    n = len(event_id)

    columns = {
        "event_id": event_id,
        "corpus": corpus,
        "doc_id": doc_id,
        "token": token,
        "token_idx": token_idx,
        "pub_year": pub_year,
        "local_window_id": local_window_id,
        "local_window_token_pos": local_window_token_pos,
        "medium_window_id": medium_window_id,
        "medium_window_token_pos": medium_window_token_pos,
        "broad_window_id": broad_window_id,
        "broad_window_token_pos": broad_window_token_pos,
    }

    for name, values in columns.items():
        if values is not None and len(values) != n:
            raise ValueError(
                f"{name} length {len(values)} != event_id length {n}"
            )

    with conn.cursor() as cur:
        with cur.copy("""
            COPY events (
                event_id,
                corpus,
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
            )
            FROM STDIN
        """) as copy:
            for i in range(n):
                copy.write_row((
                    int(event_id[i]),
                    corpus[i],
                    doc_id[i],
                    token[i],
                    int(token_idx[i]),
                    pub_year[i],
                    local_window_id[i] if local_window_id is not None else None,
                    local_window_token_pos[i] if local_window_token_pos is not None else None,
                    medium_window_id[i] if medium_window_id is not None else None,
                    medium_window_token_pos[i] if medium_window_token_pos is not None else None,
                    broad_window_id[i] if broad_window_id is not None else None,
                    broad_window_token_pos[i] if broad_window_token_pos is not None else None,
                ))