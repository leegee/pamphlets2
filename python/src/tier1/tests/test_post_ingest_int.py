from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import lancedb

from lib.corpus_db import get_connection
from lib.corpus_config import LANCE_INDEXES_DIR


SCALES = ("local", "medium", "broad")


def lance_table_name(scale: str, start: int, end: int) -> str:
    return f"{scale}__macberth__{start:04d}_{end:04d}"


def audit_postgres(conn) -> None:
    print("\nPOSTGRES")
    print("========")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events")
        event_count = cur.fetchone()[0]

        cur.execute("""
            SELECT corpus, COUNT(*)
            FROM events
            GROUP BY corpus
            ORDER BY corpus
        """)
        corpus_counts = cur.fetchall()

        cur.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE pub_year IS NULL) AS no_year,
                COUNT(DISTINCT event_id) AS distinct_ids
            FROM events
        """)
        total, no_year, distinct_ids = cur.fetchone()

        cur.execute("""
            SELECT event_id, COUNT(*)
            FROM events
            GROUP BY event_id
            HAVING COUNT(*) > 1
            LIMIT 10
        """)
        duplicate_ids = cur.fetchall()

    print(f"events:                 {event_count:,}")
    print(f"distinct event_ids:     {distinct_ids:,}")
    print(f"NULL publication years:  {no_year:,}")

    print("\nby corpus:")
    for corpus, count in corpus_counts:
        print(f"  {corpus:<8} {count:>12,}")

    if duplicate_ids:
        print("\nERROR: duplicate event_ids:")
        for event_id, count in duplicate_ids:
            print(f"  {event_id}: {count}")
    else:
        print("\nOK: no duplicate event_ids")


def audit_lance(conn, lance_root: Path) -> None:
    print("\nLANCE")
    print("=====")

    db = lancedb.connect(str(lance_root))
    tables = db.list_tables().tables

    selected = defaultdict(list)

    for table_name in tables:
        for scale in SCALES:
            prefix = f"{scale}__macberth__"
            if table_name.startswith(prefix):
                selected[scale].append(table_name)

    all_lance_ids: dict[str, set[int]] = {}

    for scale in SCALES:
        print(f"\n{scale}:")
        tables_for_scale = sorted(selected[scale])

        if not tables_for_scale:
            print("  ERROR: no tables")
            all_lance_ids[scale] = set()
            continue

        ids: set[int] = set()
        vector_count = 0

        for table_name in tables_for_scale:
            table = db.open_table(table_name)
            arrow = table.to_arrow()

            count = arrow.num_rows
            vector_count += count

            table_ids = set(
                arrow.column("event_id").to_pylist()
            )

            duplicate_count = count - len(table_ids)

            print(
                f"  {table_name:<45} "
                f"{count:>10,} rows"
            )

            if duplicate_count:
                print(
                    f"    ERROR: {duplicate_count:,} "
                    "duplicate IDs within table"
                )

            ids.update(table_ids)

        all_lance_ids[scale] = ids

        print(f"  tables:                {len(tables_for_scale):,}")
        print(f"  rows:                  {vector_count:,}")
        print(f"  distinct event_ids:    {len(ids):,}")

    print("\nCROSS-SCALE")
    print("===========")

    local_ids = all_lance_ids["local"]
    medium_ids = all_lance_ids["medium"]
    broad_ids = all_lance_ids["broad"]

    print(f"local ∩ medium:          {len(local_ids & medium_ids):,}")
    print(f"local ∩ broad:           {len(local_ids & broad_ids):,}")
    print(f"medium ∩ broad:          {len(medium_ids & broad_ids):,}")
    print(
        f"all three:               "
        f"{len(local_ids & medium_ids & broad_ids):,}"
    )

    missing_medium = local_ids - medium_ids
    missing_broad = local_ids - broad_ids
    missing_local = medium_ids - local_ids

    if missing_medium:
        print(
            f"\nERROR: {len(missing_medium):,} "
            "local events have no medium vector"
        )

    if missing_broad:
        print(
            f"ERROR: {len(missing_broad):,} "
            "local events have no broad vector"
        )

    if missing_local:
        print(
            f"ERROR: {len(missing_local):,} "
            "medium events have no local vector"
        )

    if not missing_medium and not missing_broad and not missing_local:
        print("\nOK: Lance scales contain the same event IDs")


def audit_postgres_vs_lance(conn, lance_root: Path) -> None:
    print("\nPOSTGRES ↔ LANCE")
    print("================")

    with conn.cursor() as cur:
        cur.execute("SELECT event_id FROM events")
        postgres_ids = {row[0] for row in cur.fetchall()}

    db = lancedb.connect(str(lance_root))

    lance_ids: dict[str, set[int]] = {}

    for scale in SCALES:
        ids: set[int] = set()

        prefix = f"{scale}__macberth__"

        for table_name in db.list_tables().tables:
            if table_name.startswith(prefix):
                table = db.open_table(table_name)
                ids.update(
                    table.to_arrow()
                    .column("event_id")
                    .to_pylist()
                )

        lance_ids[scale] = ids

        missing = postgres_ids - ids
        orphaned = ids - postgres_ids

        print(f"\n{scale}")
        print(f"  PostgreSQL events:       {len(postgres_ids):,}")
        print(f"  Lance vectors:           {len(ids):,}")
        print(f"  missing from Lance:      {len(missing):,}")
        print(f"  orphaned in Lance:       {len(orphaned):,}")

        if missing:
            print(
                "  first missing IDs: "
                f"{sorted(missing)[:10]}"
            )

        if orphaned:
            print(
                "  first orphaned IDs: "
                f"{sorted(orphaned)[:10]}"
            )


def main() -> None:
    conn = get_connection()

    try:
        lance_root = Path(LANCE_INDEXES_DIR)

        audit_postgres(conn)
        audit_lance(conn, lance_root)
        audit_postgres_vs_lance(conn, lance_root)

    finally:
        conn.close()


if __name__ == "__main__":
    main()