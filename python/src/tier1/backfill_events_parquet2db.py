# tier1/backfill_events_parquet2db.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import duckdb

from lib.corpus_config import EVENTSTORE_ROOT
from lib.corpus_db import get_connection
from tier1.db_observation_backend import sync_event_id_sequence


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

BATCH_SIZE = 10_000


def discover_partitions(root: Path) -> list[Path]:
    """Return all year partitions beneath the event store in chronological order."""
    partitions = [
        path
        for path in root.glob("**/year=*")
        if path.is_dir() and path.name[5:].isdigit()
    ]

    return sorted(
        partitions,
        key=lambda path: (int(path.name[5:]), str(path)),
    )


def partition_year(partition: Path) -> int:
    """Extract the year from a Hive-style year=YYYY directory."""
    return int(partition.name[5:])


def iter_partition_rows(
    partition: Path,
    batch_size: int,
) -> Iterator[list[tuple]]:
    """
    Stream metadata rows from one Parquet partition.

    Embedding columns are deliberately excluded: the migration establishes
    PostgreSQL event identity and metadata only; Lance already owns vectors.
    """
    columns = ", ".join(EVENT_COLUMNS)

    query = f"""
        SELECT {columns}
        FROM read_parquet(?)
        ORDER BY event_id
    """

    with duckdb.connect() as db:
        result = db.execute(
            query,
            [str(partition / "*.parquet")],
        )

        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break

            yield rows


def insert_batch(conn, rows: list[tuple]) -> int:
    """
    Bulk-insert one metadata batch through the partition staging table.

    Existing event IDs are accepted only when all metadata matches. New
    events are inserted in one set-based operation.
    """
    if not rows:
        return 0

    with conn.cursor() as cur:
        with cur.copy("""
            COPY event_backfill_stage (
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
            for row in rows:
                copy.write_row(row)

        cur.execute("""
            SELECT s.event_id
            FROM event_backfill_stage s
            JOIN events e
              ON e.event_id = s.event_id
            WHERE (
                e.corpus,
                e.doc_id,
                e.token,
                e.token_idx,
                e.pub_year,
                e.local_window_id,
                e.local_window_token_pos,
                e.medium_window_id,
                e.medium_window_token_pos,
                e.broad_window_id,
                e.broad_window_token_pos
            ) IS DISTINCT FROM (
                s.corpus,
                s.doc_id,
                s.token,
                s.token_idx,
                s.pub_year,
                s.local_window_id,
                s.local_window_token_pos,
                s.medium_window_id,
                s.medium_window_token_pos,
                s.broad_window_id,
                s.broad_window_token_pos
            )
            LIMIT 1
        """)

        conflict = cur.fetchone()

        if conflict is not None:
            raise RuntimeError(
                f"event_id {conflict[0]} already exists with different metadata"
            )

        cur.execute("""
            INSERT INTO events (
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
            SELECT
                s.event_id,
                s.corpus,
                s.doc_id,
                s.token,
                s.token_idx,
                s.pub_year,
                s.local_window_id,
                s.local_window_token_pos,
                s.medium_window_id,
                s.medium_window_token_pos,
                s.broad_window_id,
                s.broad_window_token_pos
            FROM event_backfill_stage s
            WHERE NOT EXISTS (
                SELECT 1
                FROM events e
                WHERE e.event_id = s.event_id
            )
        """)

        inserted = cur.rowcount

        cur.execute("TRUNCATE event_backfill_stage")

        return inserted


def backfill_partition(
    conn,
    partition: Path,
    batch_size: int,
) -> tuple[int, int]:
    year = partition_year(partition)
    source_count = 0
    inserted_count = 0

    print(f"[backfill] {partition}: starting")

    with conn.transaction():
        with conn.cursor() as cur:
            # A previous failed transaction can leave the temporary relation
            # alive on the reused PostgreSQL connection.
            cur.execute("""
                DROP TABLE IF EXISTS event_backfill_stage;

                CREATE TEMP TABLE event_backfill_stage (
                    event_id BIGINT NOT NULL,
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
                    broad_window_token_pos INTEGER
                ) ON COMMIT DROP;
            """)

        for rows in iter_partition_rows(partition, batch_size):
            source_count += len(rows)
            inserted_count += insert_batch(conn, rows)

            if source_count % (batch_size * 10) == 0:
                print(
                    f"[backfill] {partition}: "
                    f"{source_count:,} rows read, "
                    f"{inserted_count:,} inserted"
                )

    print(
        f"[backfill] {partition}: complete — "
        f"{source_count:,} rows read, "
        f"{inserted_count:,} inserted"
    )

    return source_count, inserted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Tier 1 event metadata from Parquet into PostgreSQL."
    )

    parser.add_argument(
        "--year",
        type=int,
        action="append",
        dest="years",
        help="Import only this year. May be specified more than once.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Rows fetched from DuckDB at a time (default: {BATCH_SIZE}).",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    partitions = discover_partitions(EVENTSTORE_ROOT)

    if not partitions:
        raise RuntimeError(
            f"No year partitions found beneath {EVENTSTORE_ROOT}"
        )

    if args.years:
        wanted = set(args.years)

        partitions = [
            partition
            for partition in partitions
            if partition_year(partition) in wanted
        ]

        found_years = {
            partition_year(partition)
            for partition in partitions
        }

        missing = wanted - found_years

        if missing:
            raise RuntimeError(
                "Requested year partitions not found: "
                + ", ".join(str(year) for year in sorted(missing))
            )

    print(f"[backfill] {len(partitions)} partition(s) selected")

    total_source = 0
    total_inserted = 0

    with get_connection() as conn:
        for partition in partitions:
            source_count, inserted_count = backfill_partition(
                conn,
                partition,
                args.batch_size,
            )

            total_source += source_count
            total_inserted += inserted_count

        sync_event_id_sequence(conn)

    print(
        f"[backfill] complete — "
        f"{total_source:,} rows read, "
        f"{total_inserted:,} inserted"
    )


if __name__ == "__main__":
    main()