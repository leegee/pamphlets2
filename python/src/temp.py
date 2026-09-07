from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection


TABLE_PATTERN = re.compile(
    r"^(?P<scale>local|medium|broad)"
    r"__(?P<model>macberth)"
    r"__(?P<year_start>\d{4})_(?P<year_end>\d{4})\.lance$"
)

SCALES = ("local", "medium", "broad")
BATCH_SIZE = 100_000


def discover_physical_tables() -> dict[str, list[Path]]:
    tables: dict[str, list[Path]] = defaultdict(list)

    for path in LANCE_INDEXES_DIR.glob("*.lance"):
        match = TABLE_PATTERN.match(path.name)
        if not match:
            continue

        tables[match.group("scale")].append(path)

    for scale in SCALES:
        tables[scale].sort()

    return dict(tables)


def load_postgres_event_ids(conn) -> set[int]:
    print("[postgres] loading event IDs...")

    started = time.perf_counter()
    ids: set[int] = set()

    with conn.cursor(name="events_coverage_cursor") as cur:
        cur.itersize = BATCH_SIZE
        cur.execute("SELECT event_id FROM events ORDER BY event_id")

        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break

            ids.update(int(row[0]) for row in rows)

    elapsed = time.perf_counter() - started

    print(
        f"[postgres] {len(ids):,} unique event IDs "
        f"({elapsed:.2f}s)"
    )

    return ids


def read_lance_ids(path: Path) -> tuple[set[int], int]:
    # Open the physical dataset directly; the Lance catalog may not list
    # every on-disk dataset even though the .lance directory is valid.
    db = lancedb.connect(str(path.parent))
    table = db.open_table(path.stem)

    arrow_table = table.to_arrow()
    values = arrow_table.column("event_id").to_pylist()

    return {int(value) for value in values}, len(values)

def reconcile_scale(
    scale: str,
    paths: list[Path],
    postgres_ids: set[int],
) -> bool:
    print()
    print(f"[{scale}] physical Lance tables: {len(paths)}")

    all_ids: set[int] = set()
    total_rows = 0
    duplicate_rows = 0

    started = time.perf_counter()

    for path in paths:
        ids, row_count = read_lance_ids(path)

        duplicates = row_count - len(ids)
        duplicate_rows += duplicates
        total_rows += row_count

        all_ids.update(ids)

        print(
            f"       {path.name}: "
            f"{row_count:,} rows, "
            f"{len(ids):,} unique IDs"
        )

        if duplicates:
            print(
                f"       [FAIL] {duplicates:,} duplicate event IDs"
            )

    missing = postgres_ids - all_ids
    extra = all_ids - postgres_ids

    elapsed = time.perf_counter() - started

    print()
    print(f"       Lance rows:       {total_rows:,}")
    print(f"       Lance unique IDs: {len(all_ids):,}")
    print(f"       PostgreSQL IDs:   {len(postgres_ids):,}")
    print(f"       Missing:           {len(missing):,}")
    print(f"       Extra:             {len(extra):,}")
    print(f"       Duplicate rows:    {duplicate_rows:,}")
    print(f"       elapsed:            {elapsed:.2f}s")

    if missing:
        print("\n       [FAIL] sample IDs missing from Lance:")
        for event_id in sorted(missing)[:20]:
            print(f"              {event_id}")

    if extra:
        print("\n       [FAIL] sample IDs present only in Lance:")
        for event_id in sorted(extra)[:20]:
            print(f"              {event_id}")

    ok = (
        total_rows == len(all_ids)
        and duplicate_rows == 0
        and all_ids == postgres_ids
    )

    if ok:
        print(f"[ok]   {scale}: complete coverage")
    else:
        print(f"[FAIL] {scale}: coverage mismatch")

    return ok


def main() -> int:
    print("[lance] physical observation coverage check")
    print(f"[lance] root: {LANCE_INDEXES_DIR}")

    started = time.perf_counter()

    tables = discover_physical_tables()

    print("\n[lance] physical tables discovered")

    for scale in SCALES:
        paths = tables.get(scale, [])
        print(f"       {scale}: {len(paths)}")

        for path in paths:
            print(f"              {path.name}")

    missing_scales = [
        scale
        for scale in SCALES
        if len(tables.get(scale, [])) != 10
    ]

    if missing_scales:
        print(
            "\n[FAIL] expected 10 physical tables per scale; "
            f"problem with: {', '.join(missing_scales)}"
        )
        return 1

    conn = get_connection()

    try:
        postgres_ids = load_postgres_event_ids(conn)

        results = {}

        for scale in SCALES:
            results[scale] = reconcile_scale(
                scale,
                tables[scale],
                postgres_ids,
            )

    finally:
        conn.close()

    elapsed = time.perf_counter() - started

    print()
    print("=" * 72)

    if all(results.values()):
        print("[COMPLETE] ALL Lance scales have complete event coverage.")
        print()
        print("PostgreSQL == local == medium == broad")
        print()
        print("Every PostgreSQL observation has exactly one Lance row")
        print("at each of the three contextual scales.")
        print()
        print("Parquet is redundant for vector storage.")
    else:
        print("[COMPLETE] coverage check FAILED")
        print()
        print("DO NOT delete the Parquet store.")

    print(f"\nElapsed: {elapsed:.2f}s")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
