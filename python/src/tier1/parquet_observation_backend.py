"""
parquet_observation_backend.py — Parquet + DuckDB backend for the observation API

Implements ObservationWriter, ObservationStream and ObservationLookup
against a hive-partitioned Parquet layout, with DuckDB for predicate
pushdown and batched reads. This is the only registered backend; Zarr
did not scale to current Tier 1 observation volume.

Layout
------
    <root>/
        year=1640/
            part-000000.parquet
            part-000001.parquet
        year=1641/
            ...

Each part file contains the full observation schema (metadata + three
embedding scales). Partitioning by pub_year matches the year_filter path
used by FAISS builds and diachronic queries.

Embeddings are stored as Arrow fixed-size lists of float32 so DuckDB can
project them without decoding the entire row group when only metadata is
needed.

TODO
----
Enforce a clear separation of concerns.

Usage
-----
    from tier1.observation_store_api import (
        open_observation_writer,
        open_observation_stream,
        open_observation_lookup,
    )
    # import registers the backend
    import tier1.parquet_observation_backend  #

    writer = open_observation_writer(root, dim=768)
    stream = open_observation_stream(root)
    lookup = open_observation_lookup(root)
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tier1.observation_store_api import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    NO_WINDOW_TOKEN_POS,
    SCALES,
)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _embedding_type(dim: int) -> pa.DataType:
    return pa.list_(pa.float32(), dim)


def observation_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("event_id", pa.int64()),
            ("corpus", pa.string()),
            ("doc_id", pa.string()),
            ("token", pa.string()),
            ("token_idx", pa.int64()),
            ("pub_year", pa.int16()),
            ("local_window_id", pa.int64()),
            ("local_window_token_pos", pa.int32()),
            ("medium_window_id", pa.int64()),
            ("medium_window_token_pos", pa.int32()),
            ("broad_window_id", pa.int64()),
            ("broad_window_token_pos", pa.int32()),
            ("emb_local", _embedding_type(dim)),
            ("emb_medium", _embedding_type(dim)),
            ("emb_broad", _embedding_type(dim)),
        ]
    )


def _glob(root: Path) -> str:
    """DuckDB-friendly recursive glob over all part files."""
    # hive-style year=*/part-*.parquet plus any loose *.parquet at root
    return str(root / "**" / "*.parquet")


def _list_parquet_files(root: Path) -> list[str]:
    """Walk the tree once and return a sorted list of part file paths."""
    return sorted(str(p) for p in root.rglob("*.parquet"))


def _files_sql_literal(files: Sequence[str]) -> str:
    """
    Render a pre-enumerated file list as a DuckDB array literal, e.g.
    ['a.parquet','b.parquet'], so read_parquet() can be called against a
    fixed list instead of re-walking the directory tree (a glob pattern
    string) on every single query.
    """
    escaped = ",".join(
        "'" + f.replace("'", "''") + "'" for f in files
    )
    return f"[{escaped}]"


def _has_parquet(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.rglob("*.parquet"))


# ---------------------------------------------------------------------------
# Parquet write tuning (Tier 1 is write-once, read-many)
# ---------------------------------------------------------------------------
#
# Workload: string metadata (high dict benefit) + three float32 embedding
# columns (modest zstd gain, dominate size). Prefer stronger compression
# and moderate row groups so DuckDB can skip unread row groups without
# decoding multi-hundred-MB blocks into RAM.
#
# Defaults chosen for EEBO-scale offline builds on workstation SSDs:
#   zstd level 6  — clearly smaller than level 3, still fast enough for
#                   Tier 1 (embedding time dominates, not Parquet I/O)
#   row_group_size 16_384 — ~16k events × 3 × 768 × 4 ≈ 150 MB uncompressed
#                   embeddings per group; comfortable for 64 GB hosts
#   data_page_size 1 MiB — fewer pages than the tiny default under large
#                   fixed-size lists, better sequential scan throughput
#   dictionary on corpus/doc_id/token only — embeddings stay plain

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 6
PARQUET_ROW_GROUP_SIZE = 16_384
PARQUET_DATA_PAGE_SIZE = 1 << 20  # 1 MiB
PARQUET_DICT_COLUMNS = ("corpus", "doc_id", "token")


def write_observation_parquet(
    table: pa.Table,
    path: str | Path,
    *,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    data_page_size: int = PARQUET_DATA_PAGE_SIZE,
    use_dictionary: Sequence[str] | bool | None = None,
) -> None:
    """
    Write an observation table with project-standard Parquet encoding.

    Used by ParquetObservationWriter and compact_parquet_parts so online
    appends and offline compaction stay consistent.
    """
    if use_dictionary is None:
        use_dictionary = list(PARQUET_DICT_COLUMNS)
    path = Path(path)
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        data_page_size=data_page_size,
        use_dictionary=use_dictionary,
        write_statistics=True,
        # Column order matches schema; statistics enable year/token filters.
        coerce_timestamps="us",
    )


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # Prefer threads for scan; callers may run many sequential queries.
    con.execute("SET threads TO 4")
    return con


def _probe_dim(con: duckdb.DuckDBPyConnection, files_sql: str) -> int:
    """
    Infer embedding dimensionality from whichever scale column has data.

    A store may only ever have had a subset of scales written (e.g. a
    run with --scales medium leaves emb_local/emb_broad null for every
    row), so this can't assume any single column is populated. It scans
    for the first row where at least one scale is non-null and reads
    that column's length.

    `files_sql` is a pre-built DuckDB list literal, for example
    "['a.parquet','b.parquet']" (see _files_sql_literal), not a bare
    glob pattern — it must be interpolated as-is, with no surrounding
    quotes, or DuckDB parses the whole bracketed literal as one string.
    """
    row = con.execute(
        f"""
        SELECT emb_local, emb_medium, emb_broad
        FROM read_parquet({files_sql}, hive_partitioning=true, union_by_name=true)
        WHERE emb_local IS NOT NULL
           OR emb_medium IS NOT NULL
           OR emb_broad IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return 0
    for v in row:
        if v is not None:
            return len(v)
    return 0


def _validate_scales(scales: Sequence[str]) -> tuple[str, ...]:
    """
    Normalise a requested scale sequence: check membership in SCALES,
    drop duplicates, preserve caller order. Raises ValueError on an
    unknown scale name so a typo fails loudly instead of silently
    reading nothing.
    """
    seen: list[str] = []
    for s in scales:
        if s not in SCALES:
            raise ValueError(f"Unknown scale {s!r}; must be one of {SCALES}")
        if s not in seen:
            seen.append(s)
    return tuple(seen)



# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

# Default part sizing: avoid one tiny file per document.
# ~100k rows × 3 × 768 float32 ≈ 0.9 GB uncompressed; zstd is much smaller.
DEFAULT_MIN_ROWS = 100_000
DEFAULT_MIN_BYTES = 256 * 1024 * 1024  # 256 MiB estimated uncompressed payload


class ParquetObservationWriter:
    """
    Append-only writer with per-year row buffers.

    Each row is one contextual observation identified by event_id.
    The three embeddings belong to that observation but may have
    different source windows, so window provenance is stored per scale.

    event_id is the only observation identity. There is deliberately no
    concept_id or vector_id: an observation is not a concept and there is
    no single vector identity for an ensemble of three vectors.

    Partial-scale writes
    ---------------------
    append_events accepts each scale's (emb_<scale>, <scale>_window_id,
    <scale>_window_token_pos) as an optional group. A run whose --scales
    only covers a subset (e.g. just "medium") omits the other groups; the
    corresponding columns are written as null for that batch. A later run
    that computes "local" for the same store just calls append_events
    with local's group filled in — nothing needs to be rewritten. Readers
    (ParquetObservationLookup / ParquetObservationStream) raise a clear
    error if asked for a scale that turns out to be null for a given row,
    rather than silently returning zeros.

    Note on file listing
    ---------------------
    Unlike the read-only Lookup/Stream classes, this writer's file list
    is deliberately *not* cached: n_events/get_doc_keys/get_event_ids
    must see part files written earlier in the same process (and by
    other writers), so each call re-walks the directory. This is fine
    since these are low-frequency, whole-store administrative queries,
    not the per-batch embedding fetch path.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dim: int,
        min_rows: int = DEFAULT_MIN_ROWS,
        min_bytes: int = DEFAULT_MIN_BYTES,
        compression: str = PARQUET_COMPRESSION,
        compression_level: int = PARQUET_COMPRESSION_LEVEL,
        row_group_size: int = PARQUET_ROW_GROUP_SIZE,
        data_page_size: int = PARQUET_DATA_PAGE_SIZE,
        **_kwargs: Any,
    ):
        self.root = Path(path)
        self.dim = int(dim)
        self.min_rows = max(1, int(min_rows))
        self.min_bytes = max(0, int(min_bytes))
        self.compression = compression
        self.compression_level = int(compression_level)
        self.row_group_size = max(1, int(row_group_size))
        self.data_page_size = max(64 * 1024, int(data_page_size))
        self._schema = observation_schema(self.dim)
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._n_events: int | None = None
        self._con = _connect()
        self._buffers: dict[int, list[pa.Table]] = {}
        self._buffer_rows: dict[int, int] = {}
        self._buffer_bytes: dict[int, int] = {}
        self._closed = False

    def _estimate_bytes(self, n: int, n_scales: int = 3) -> int:
        emb = n * self.dim * 4 * n_scales
        meta = n * 96
        return emb + meta

    def append_events(
        self,
        *,
        event_id,
        corpus,
        doc_id,
        token,
        token_idx,
        pub_year,
        local_window_id=None,
        local_window_token_pos=None,
        medium_window_id=None,
        medium_window_token_pos=None,
        broad_window_id=None,
        broad_window_token_pos=None,
        emb_local=None,
        emb_medium=None,
        emb_broad=None,
    ) -> None:
        if self._closed:
            raise RuntimeError("ParquetObservationWriter is closed")

        event_id = np.asarray(event_id, dtype=np.int64)
        corpus = np.asarray(corpus).astype(str)
        doc_id = np.asarray(doc_id).astype(str)
        token = np.asarray(token).astype(str)
        token_idx = np.asarray(token_idx, dtype=np.int64)
        pub_year = np.asarray(pub_year, dtype=np.int16)

        n = len(event_id)

        if n == 0:
            return

        for name, arr in (
            ("corpus", corpus),
            ("doc_id", doc_id),
            ("token", token),
            ("token_idx", token_idx),
            ("pub_year", pub_year),
        ):
            if len(arr) != n:
                raise ValueError( f"{name} length {len(arr)} != event_id length {n}" )

        # Each scale's (emb_<scale>, <scale>_window_id,
        # <scale>_window_token_pos) is an all-or-nothing group. A run that
        # only computed a subset of scales (e.g. --scales medium) omits
        # the other scales' groups entirely; those columns are written as
        # null for every row in this call rather than raising, so a
        # single-scale run and a later run adding another scale can both
        # append to the same store.
        raw_scale_args = {
            "local": (emb_local, local_window_id, local_window_token_pos),
            "medium": (emb_medium, medium_window_id, medium_window_token_pos),
            "broad": (emb_broad, broad_window_id, broad_window_token_pos),
        }

        present: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for scale, (emb, wid, wpos) in raw_scale_args.items():
            supplied = (emb is not None, wid is not None, wpos is not None)
            if not any(supplied):
                continue
            if not all(supplied):
                raise ValueError(
                    f"scale {scale!r}: emb_{scale}, {scale}_window_id and "
                    f"{scale}_window_token_pos must be supplied together "
                    f"(or all omitted) — got emb={supplied[0]}, "
                    f"window_id={supplied[1]}, window_token_pos={supplied[2]}"
                )

            emb = np.asarray(emb, dtype=np.float32)
            wid = np.asarray(wid, dtype=np.int64)
            wpos = np.asarray(wpos, dtype=np.int32)

            if emb.shape != (n, self.dim):
                raise ValueError( f"emb_{scale} shape {emb.shape} != ({n}, {self.dim})" )

            if len(wid) != n:
                raise ValueError( f"{scale}_window_id length {len(wid)} != {n}" )

            if len(wpos) != n:
                raise ValueError( f"{scale}_window_token_pos length {len(wpos)} != {n}" )

            present[scale] = (emb, wid, wpos)

        if not present:
            raise ValueError(
                "append_events requires at least one scale's "
                "(emb_<scale>, <scale>_window_id, <scale>_window_token_pos) group; "
                "got none"
            )

        years = np.unique(pub_year)

        with self._lock:
            for year in years:
                mask = pub_year == year

                table = self._build_table(
                    event_id[mask],
                    corpus[mask],
                    doc_id[mask],
                    token[mask],
                    token_idx[mask],
                    pub_year[mask],
                    {
                        scale: (emb[mask], wid[mask], wpos[mask])
                        for scale, (emb, wid, wpos) in present.items()
                    },
                )

                y = int(year)
                nrows = table.num_rows

                self._buffers.setdefault(y, []).append(table)
                self._buffer_rows[y] = (
                    self._buffer_rows.get(y, 0) + nrows
                )
                self._buffer_bytes[y] = (
                    self._buffer_bytes.get(y, 0)
                    + self._estimate_bytes(nrows, len(present))
                )

                if (
                    self._buffer_rows[y] >= self.min_rows
                    or self._buffer_bytes[y] >= self.min_bytes
                ):
                    self._flush_year_unlocked(y)

            self._n_events = None

    def flush(self) -> None:
        with self._lock:
            for year in list(self._buffers):
                self._flush_year_unlocked(year)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _flush_year_unlocked(self, year: int) -> None:
        tables = self._buffers.pop(year, None)
        self._buffer_rows.pop(year, None)
        self._buffer_bytes.pop(year, None)

        if not tables:
            return

        table = (
            pa.concat_tables(tables)
            if len(tables) > 1
            else tables[0]
        )

        self._write_part(year, table)
        self._n_events = None

    def _build_table(
        self,
        event_id,
        corpus,
        doc_id,
        token,
        token_idx,
        pub_year,
        scales: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> pa.Table:
        n = len(event_id)

        def emb_col(mat: np.ndarray) -> pa.Array:
            flat = pa.array( mat.reshape(-1), type=pa.float32() )
            return pa.FixedSizeListArray.from_arrays(
                flat,
                self.dim,
            )

        def scale_columns(name: str) -> tuple[pa.Array, pa.Array, pa.Array]:
            triple = scales.get(name)
            if triple is not None:
                emb, wid, wpos = triple
                return (
                    emb_col(emb),
                    pa.array(wid, type=pa.int64()),
                    pa.array(wpos, type=pa.int32()),
                )
            # This scale wasn't computed for this batch: write null so the
            # row still matches the fixed schema. Null is distinct from
            # NO_WINDOW_TOKEN_POS (-1) — null means "this scale was never
            # computed here", -1 means "computed, but the token fell
            # outside this scale's window".
            return (
                pa.nulls(n, type=_embedding_type(self.dim)),
                pa.nulls(n, type=pa.int64()),
                pa.nulls(n, type=pa.int32()),
            )

        emb_local_col, local_wid_col, local_wpos_col = scale_columns("local")
        emb_medium_col, medium_wid_col, medium_wpos_col = scale_columns("medium")
        emb_broad_col, broad_wid_col, broad_wpos_col = scale_columns("broad")

        return pa.table(
            {
                "event_id": pa.array(event_id, type=pa.int64()),
                "corpus": pa.array(corpus, type=pa.string()),
                "doc_id": pa.array(doc_id, type=pa.string()),
                "token": pa.array(token, type=pa.string()),
                "token_idx": pa.array(token_idx, type=pa.int64()),
                "pub_year": pa.array(pub_year, type=pa.int16()),
                "local_window_id": local_wid_col,
                "local_window_token_pos": local_wpos_col,
                "medium_window_id": medium_wid_col,
                "medium_window_token_pos": medium_wpos_col,
                "broad_window_id": broad_wid_col,
                "broad_window_token_pos": broad_wpos_col,
                "emb_local": emb_local_col,
                "emb_medium": emb_medium_col,
                "emb_broad": emb_broad_col,
            },
            schema=self._schema,
        )

    def _write_part(self, year: int, table: pa.Table) -> None:
        if table.num_rows == 0:
            return

        part_dir = self.root / f"year={year}"
        part_dir.mkdir(parents=True, exist_ok=True)

        part_name = f"part-{uuid.uuid4().hex[:12]}.parquet"
        dest = part_dir / part_name
        tmp = part_dir / f".{part_name}.tmp"

        write_observation_parquet(
            table,
            tmp,
            compression=self.compression,
            compression_level=self.compression_level,
            row_group_size=self.row_group_size,
            data_page_size=self.data_page_size,
        )

        tmp.rename(dest)

    def _count(self) -> int:
        if not _has_parquet(self.root):
            return 0

        glob = _glob(self.root)

        row = self._con.execute(
            f"""
            SELECT count(*)
            FROM read_parquet(
                '{glob}',
                hive_partitioning=true,
                union_by_name=true
            )
            """
        ).fetchone()

        return int(row[0]) if row else 0

    @property
    def n_events(self) -> int:
        if self._n_events is None:
            self._n_events = self._count()

        buffered = sum(self._buffer_rows.values())
        return self._n_events + buffered

    def get_doc_keys(self) -> set[tuple[str, str]]:
        if not _has_parquet(self.root):
            return set()

        glob = _glob(self.root)

        rows = self._con.execute(
            f"""
            SELECT DISTINCT corpus, doc_id
            FROM read_parquet(
                '{glob}',
                hive_partitioning=true,
                union_by_name=true
            )
            """
        ).fetchall()

        return {(str(c), str(d)) for c, d in rows}

    def get_event_ids(self) -> set[int]:
        if not _has_parquet(self.root):
            return set()

        glob = _glob(self.root)

        rows = self._con.execute(
            f"""
            SELECT event_id
            FROM read_parquet(
                '{glob}',
                hive_partitioning=true,
                union_by_name=true
            )
            """
        ).fetchall()

        return {int(r[0]) for r in rows}

    def embedding_dim(self) -> int:
        return self.dim

    def __len__(self) -> int:
        return self.n_events



# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class ParquetObservationStream:
    """
    Batch streaming of multi-scale embeddings via DuckDB.

    Does not materialise the full corpus. year_filter is pushed into the
    SQL WHERE clause so unused year partitions are skipped when hive
    partitioning is present.

    The part-file list is enumerated once at construction (see
    ParquetObservationLookup for why) rather than re-globbed on every
    query.
    """

    def __init__(self, root: str | Path, **_kwargs: Any):
        self.root = Path(root)
        self._con = _connect()
        self._dim: int | None = None
        self._files = _list_parquet_files(self.root)
        self._files_sql = _files_sql_literal(self._files) if self._files else None

    def _ensure_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        if not self._files_sql:
            self._dim = 0
            return 0
        self._dim = _probe_dim(self._con, self._files_sql)
        return self._dim

    def iter_multi_scale_embeddings(
        self,
        batch_size: int = 8192,
        year_filter: Optional[set[int]] = None,
        year_manifest: Optional[Mapping[Any, np.ndarray]] = None,
        scales: Sequence[str] = SCALES,
    ) -> Iterator[
        tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            np.ndarray,
            np.ndarray,
        ]
    ]:
        scales = _validate_scales(scales)

        if not self._files_sql:
            return
            yield  # make this a generator  # noqa: unreachable

        dim = self._ensure_dim()

        where = ""
        params: list[Any] = []
        if year_filter is not None and len(year_filter) > 0:
            # DuckDB IN list
            placeholders = ",".join("?" for _ in year_filter)
            # where = f"WHERE pub_year IN ({placeholders})"
            where = f"WHERE year IN ({placeholders})"
            params = list(year_filter)

        # Only the requested scales' embedding columns go into the SELECT
        # list, so DuckDB never decodes the fixed-size-list columns for
        # scales the caller didn't ask for.
        emb_cols = ", ".join(f"emb_{s}" for s in scales)
        sql = f"""
            SELECT {emb_cols}, event_id, pub_year
            FROM read_parquet({self._files_sql}, hive_partitioning=true, union_by_name=true)
            {where}
        """
        # Use a fresh connection for the streaming cursor so concurrent
        # year_bounds / other queries on self._con stay safe.
        con = _connect()
        try:
            con.execute(sql, params)
            n_scale_cols = len(scales)
            while True:
                rows = con.fetchmany(batch_size)
                if not rows:
                    break
                n = len(rows)
                mats = {s: np.empty((n, dim), dtype=np.float32) for s in scales}
                eids = np.empty(n, dtype=np.int64)
                years = np.empty(n, dtype=np.int16)
                for i, row in enumerate(rows):
                    for j, s in enumerate(scales):
                        val = row[j]
                        if val is None:
                            raise ValueError(
                                f"scale {s!r} is null for event_id={row[n_scale_cols]} "
                                f"— this store has rows where {s!r} was never "
                                f"computed (a run with --scales that excluded "
                                f"it). Filter to a year/subset that has {s!r} "
                                f"fully populated, or request a different scale."
                            )
                        mats[s][i] = val
                    eids[i] = row[n_scale_cols]
                    years[i] = row[n_scale_cols + 1]
                yield (
                    mats.get("local"),
                    mats.get("medium"),
                    mats.get("broad"),
                    eids,
                    years,
                )
        finally:
            con.close()

    def year_bounds(self) -> tuple[int, int]:
        if not self._files_sql:
            raise RuntimeError("No parquet data found in store")
        row = self._con.execute(
            f"""
            SELECT min(pub_year), max(pub_year)
            FROM read_parquet({self._files_sql}, hive_partitioning=true, union_by_name=true)
            """
        ).fetchone()
        if row is None or row[0] is None:
            raise RuntimeError("No pub_year data found in any parquet file")
        return int(row[0]), int(row[1])




# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

class ParquetObservationLookup:
    """
    Selective observation access backed by Parquet + DuckDB.

    Metadata strategy
    -----------------
    On construction, metadata columns (no embeddings) are loaded into
    compact NumPy struct-of-arrays. Peak RSS is dominated by metadata,
    not the ~16 GiB embedding matrices.

    Optional constructor kwargs:
        forms: iterable of tokens — only rows matching these tokens are
               loaded (case-insensitive). Use for single-concept runs.
        false_positives: tokens to exclude when forms is set.
        years: iterable of pub_year values to restrict the load.

    Embedding strategy
    ------------------
    1. If attach_index() was called, vectors are reconstructed from the
       attached per-year FAISS indices.
    2. Otherwise vectors are fetched on demand from Parquet via DuckDB
       keyed by event_id (batched), one scale at a time. Selective
       single-scale reads (get_scale_embedding / get_scale_embeddings)
       and the multi-scale ensemble methods (get_ensemble_embedding /
       get_embeddings / get_concatenated_embeddings, via their `scales`
       argument) both route through the same per-scale fetch path, so
       a scale that was never requested is never read from Parquet.

    File listing
    ------------
    A ParquetObservationLookup is opened once per run and is read-only
    from the caller's point of view (Tier 1 writers are a separate
    process). The part-file list is therefore enumerated exactly once,
    at construction, and reused as a DuckDB array literal for every
    query. Previously each embedding fetch re-passed a glob pattern
    ('root/**/*.parquet') to read_parquet(), which forced DuckDB to
    re-walk the whole hive-partitioned directory tree on every call —
    including calls fetching only a handful of rows. That per-call
    directory walk, multiplied across many small batches, dominated
    Tier 2 runtime far more than the actual Parquet decode cost.
    """

    _META_SQL_COLS = (
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

    def __init__(
        self,
        root: str | Path,
        *,
        forms: Optional[Sequence[str]] = None,
        false_positives: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
        **_kwargs: Any,
    ):
        self.root = Path(root)
        self._con = _connect()
        self._index = None  # optional FAISS attach
        self._dim: int | None = None

        # Enumerated once; see class docstring "File listing".
        self._files = _list_parquet_files(self.root)
        self._files_sql = _files_sql_literal(self._files) if self._files else None

        # Cache is keyed by scale, then event_id, so a caller that only
        # ever touches "local" never pays for medium/broad cache entries,
        # and clearing one scale's cache doesn't disturb the others.
        self._emb_cache: dict[str, dict[int, np.ndarray]] = {
            s: {} for s in SCALES
        }
        self._emb_cache_max = 50_000

        self._forms = {f.lower() for f in forms} if forms else None
        self._fps = {f.lower() for f in false_positives} if false_positives else set()
        self._years = set(int(y) for y in years) if years else None

        self._pos: dict[int, int] = {}
        self._pos_by_occurrence: dict[tuple[str, str, int], list[int]] = {}
        self._load_metadata()

    # --- loading -----------------------------------------------------------

    def _load_metadata(self) -> None:
        empty = {
            "event_id": np.empty(0, dtype=np.int64),
            "corpus": np.empty(0, dtype=object),
            "doc_id": np.empty(0, dtype=object),
            "token": np.empty(0, dtype=object),
            "token_idx": np.empty(0, dtype=np.int64),
            "pub_year": np.empty(0, dtype=np.int16),
            "local_window_id": np.empty(0, dtype=np.int64),
            "local_window_token_pos": np.empty(
                0,
                dtype=np.int32,
            ),
            "medium_window_id": np.empty(0, dtype=np.int64),
            "medium_window_token_pos": np.empty(
                0,
                dtype=np.int32,
            ),
            "broad_window_id": np.empty(0, dtype=np.int64),
            "broad_window_token_pos": np.empty(
                0,
                dtype=np.int32,
            ),
        }
        for k, v in empty.items():
            setattr(self, k, v)

        if not self._files_sql:
            return

        clauses: list[str] = []
        params: list[Any] = []

        if self._forms is not None:
            placeholders = ",".join("?" for _ in self._forms)
            clauses.append(f"lower(token) IN ({placeholders})")
            params.extend(self._forms)
        if self._fps:
            placeholders = ",".join("?" for _ in self._fps)
            clauses.append(f"lower(token) NOT IN ({placeholders})")
            params.extend(self._fps)
        if self._years is not None:
            placeholders = ",".join("?" for _ in self._years)
            clauses.append(f"pub_year IN ({placeholders})")
            params.extend(self._years)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cols = ", ".join(self._META_SQL_COLS)
        sql = f"""
            SELECT {cols}
            FROM read_parquet({self._files_sql}, hive_partitioning=true, union_by_name=true)
            {where}
        """
        rel = self._con.execute(sql, params)
        # DuckDB may return a RecordBatchReader; materialise to a Table.
        arrow = rel.arrow()
        if hasattr(arrow, "read_all"):
            arrow = arrow.read_all()
        elif not isinstance(arrow, pa.Table):
            arrow = pa.Table.from_batches(list(arrow))

        n = arrow.num_rows
        if n == 0:
            return

        self.event_id = arrow.column("event_id").to_numpy().astype(
            np.int64,
            copy=False,
        )
        self.token_idx = arrow.column("token_idx").to_numpy().astype(
            np.int64,
            copy=False,
        )
        self.pub_year = arrow.column("pub_year").to_numpy().astype(
            np.int16,
            copy=False,
        )

        for name in (
            "local_window_id",
            "medium_window_id",
            "broad_window_id",
        ):
            # A scale that wasn't computed for this row is null here (see
            # ParquetObservationWriter's partial-scale write contract).
            # Fill with NO_WINDOW_TOKEN_POS rather than letting the null
            # -> float -> int cast silently produce garbage values.
            col = pc.fill_null(arrow.column(name), NO_WINDOW_TOKEN_POS)
            setattr(
                self,
                name,
                col.to_numpy(zero_copy_only=False).astype(
                    np.int64,
                    copy=False,
                ),
            )

        for name in (
            "local_window_token_pos",
            "medium_window_token_pos",
            "broad_window_token_pos",
        ):
            col = pc.fill_null(arrow.column(name), NO_WINDOW_TOKEN_POS)
            setattr(
                self,
                name,
                col.to_numpy(zero_copy_only=False).astype(
                    np.int32,
                    copy=False,
                ),
            )

        self.corpus = np.asarray(
            arrow.column("corpus").to_pylist(),
            dtype=object,
        )
        self.doc_id = np.asarray(
            arrow.column("doc_id").to_pylist(),
            dtype=object,
        )
        self.token = np.asarray(
            arrow.column("token").to_pylist(),
            dtype=object,
        )

        self._pos = {int(eid): i for i, eid in enumerate(self.event_id.tolist())}
        self._build_position_index()

    def _build_position_index(self) -> None:
        self._pos_by_occurrence = {}
        for corpus, doc_id, token_idx, eid in zip(
            self.corpus.tolist(),
            self.doc_id.tolist(),
            self.token_idx.tolist(),
            self.event_id.tolist(),
        ):
            key = (str(corpus), str(doc_id), int(token_idx))
            bucket = self._pos_by_occurrence.get(key)
            if bucket is None:
                self._pos_by_occurrence[key] = [eid]
            else:
                bucket.append(eid)

    # --- size / schema -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.event_id)

    @property
    def available_years(self) -> np.ndarray:
        if len(self.pub_year) == 0:
            return np.empty(0, dtype=np.int16)
        return np.unique(self.pub_year)

    # --- identity ----------------------------------------------------------

    def get_pos(self, event_id: int) -> int:
        return self._pos[int(event_id)]

    def _row_to_dict(self, pos: int) -> dict:
        def window_pos(value: int) -> int | None:
            # NO_WINDOW_TOKEN_POS (-1) means either "computed but the
            # token fell outside this scale's window" (token_pos) or
            # "this scale was never computed for this row, see
            # ParquetObservationWriter's partial-scale write contract"
            # (window_id). Either way, None reads more honestly than -1.
            return (
                None
                if value == NO_WINDOW_TOKEN_POS
                else int(value)
            )

        return {
            "event_id": int(self.event_id[pos]),
            "doc_id": str(self.doc_id[pos]),
            "corpus": str(self.corpus[pos]),
            "token": str(self.token[pos]),
            "token_idx": int(self.token_idx[pos]),
            "pub_year": int(self.pub_year[pos]),
            "local_window_id": window_pos(self.local_window_id[pos]),
            "local_window_token_pos": window_pos(
                self.local_window_token_pos[pos]
            ),
            "medium_window_id": window_pos(self.medium_window_id[pos]),
            "medium_window_token_pos": window_pos(
                self.medium_window_token_pos[pos]
            ),
            "broad_window_id": window_pos(self.broad_window_id[pos]),
            "broad_window_token_pos": window_pos(
                self.broad_window_token_pos[pos]
            ),
        }


    def get_event_metadata(self, event_id: int) -> dict:
        return self._row_to_dict(self.get_pos(event_id))

    def get_event(self, event_id: int) -> dict:
        pos = self.get_pos(event_id)
        d = self._row_to_dict(pos)
        d["embedding"] = self.get_ensemble_embedding(pos)
        return d

    # --- form / position queries -------------------------------------------

    def iter_matching_event_ids(
        self,
        forms: Sequence[str],
        false_positives: Optional[Sequence[str]] = None,
    ) -> Iterator[int]:
        forms_set = {f.lower() for f in forms}
        fps = {f.lower() for f in (false_positives or [])}
        if len(self.token) == 0:
            return
        tokens_lower = np.char.lower(self.token.astype(str))
        mask = np.isin(tokens_lower, list(forms_set))
        if fps:
            mask &= ~np.isin(tokens_lower, list(fps))
        seen: set[int] = set()
        for eid in self.event_id[mask]:
            eid = int(eid)
            if eid in seen:
                continue
            seen.add(eid)
            yield eid

    def find_matching_event_ids(
        self,
        forms: Sequence[str],
        false_positives: Optional[Sequence[str]] = None,
    ) -> list[int]:
        return list(self.iter_matching_event_ids(forms, false_positives))

    def find_event_ids_by_positions(
        self,
        positions: Sequence[tuple[str, str, int]],
    ) -> dict[tuple[str, str, int], list[int]]:
        result: dict[tuple[str, str, int], list[int]] = {}
        for corpus, doc_id, token_idx in positions:
            key = (str(corpus), str(doc_id), int(token_idx))
            if key in result:
                continue
            result[key] = list(self._pos_by_occurrence.get(key, []))
        return result

    # --- embeddings --------------------------------------------------------

    def attach_index(self, index: Any) -> None:
        """
        Attach per-year FAISS indices for lazy reconstruction.

        Expected shape: dict[year][scale] -> object with
            .reconstruct(event_id) -> (dim,) ndarray
            .reconstruct_many(event_ids) -> (n, dim) ndarray
            .dim attribute
        """
        self._index = index
        if index:
            any_year = next(iter(index.values()))
            self._dim = next(iter(any_year.values())).dim


    def _ensure_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        if not self._files_sql:
            self._dim = 0
            return 0
        self._dim = _probe_dim(self._con, self._files_sql)
        return self._dim


    def _fetch_scale_from_parquet(
        self, event_ids: Sequence[int], scale: str
    ) -> np.ndarray:
        """
        (n, dim) matrix for one scale, aligned to event_ids order.

        Only the emb_{scale} column is named in the SELECT list — the
        other two scales' embedding data is never read from Parquet by
        this call. Uses an in-process cache keyed by (scale, event_id).

        Reuses self._files_sql (enumerated once at construction) instead
        of re-globbing the store on every call — see class docstring.
        """
        dim = self._ensure_dim()
        n = len(event_ids)
        out = np.empty((n, dim), dtype=np.float32)
        cache = self._emb_cache[scale]

        missing: list[int] = []
        missing_idx: list[int] = []
        for i, eid in enumerate(event_ids):
            eid = int(eid)
            cached = cache.get(eid)
            if cached is not None:
                out[i] = cached
            else:
                missing.append(eid)
                missing_idx.append(i)

        if missing:
            placeholders = ",".join("?" for _ in missing)
            col = f"emb_{scale}"
            sql = f"""
                SELECT event_id, {col}
                FROM read_parquet({self._files_sql}, hive_partitioning=true, union_by_name=true)
                WHERE event_id IN ({placeholders})
            """
            rows = self._con.execute(sql, missing).fetchall()
            by_id = {int(r[0]): r[1] for r in rows}

            # Evict this scale's cache if oversized (simple FIFO-ish clear).
            # Other scales' caches are untouched.
            if len(cache) + len(by_id) > self._emb_cache_max:
                cache.clear()

            for i, eid in zip(missing_idx, missing):
                if eid not in by_id:
                    raise KeyError(f"event_id={eid} not found in parquet store")
                vec = by_id[eid]
                if vec is None:
                    raise KeyError(
                        f"scale {scale!r} was not computed for event_id={eid} "
                        f"(embedding is null in the store — this event was "
                        f"written by a run whose --scales didn't include "
                        f"{scale!r})"
                    )
                arr = np.asarray(vec, dtype=np.float32)
                out[i] = arr
                cache[eid] = arr

        return out


    def _fetch_scales_from_parquet(
        self,
        event_ids: Sequence[int],
        scales: Sequence[str],
    ) -> dict[str, np.ndarray]:
        """
        Fetch several embedding scales for the same event IDs in one Parquet
        scan and return each scale aligned to the caller's event_id order.

        The event metadata is already resident in memory, so publication years
        can be used to prune the hive-partitioned Parquet tree. Fetching all
        requested scales together also avoids scanning the observation store
        once per scale.
        """
        scales = _validate_scales(scales)

        if not scales:
            return {}

        event_ids = [int(eid) for eid in event_ids]
        n = len(event_ids)

        dim = self._ensure_dim()

        outputs = {
            scale: np.empty((n, dim), dtype=np.float32)
            for scale in scales
        }

        if not event_ids:
            return outputs

        missing_by_scale: dict[str, list[int]] = {
            scale: [] for scale in scales
        }
        missing_positions_by_scale: dict[str, list[int]] = {
            scale: [] for scale in scales
        }

        for position, eid in enumerate(event_ids):
            for scale in scales:
                cached = self._emb_cache[scale].get(eid)
                if cached is not None:
                    outputs[scale][position] = cached
                else:
                    missing_by_scale[scale].append(eid)
                    missing_positions_by_scale[scale].append(position)

        scales_to_fetch = [
            scale
            for scale in scales
            if missing_by_scale[scale]
        ]

        if not scales_to_fetch:
            return outputs

        missing_ids = sorted({
            eid
            for scale in scales_to_fetch
            for eid in missing_by_scale[scale]
        })

        positions = [
            self.get_pos(eid)
            for eid in missing_ids
        ]
        years = sorted({
            int(self.pub_year[pos])
            for pos in positions
        })

        id_placeholders = ",".join("?" for _ in missing_ids)
        year_placeholders = ",".join("?" for _ in years)

        embedding_columns = ", ".join(
            f"emb_{scale}"
            for scale in scales_to_fetch
        )

        sql = f"""
            SELECT event_id, {embedding_columns}
            FROM read_parquet(
                {self._files_sql},
                hive_partitioning=true,
                union_by_name=true
            )
            WHERE year IN ({year_placeholders})
            AND event_id IN ({id_placeholders})
        """

        rows = self._con.execute(
            sql,
            [*years, *missing_ids],
        ).fetchall()

        by_id = {
            int(row[0]): row[1:]
            for row in rows
        }

        if len(by_id) != len(missing_ids):
            missing = [
                eid
                for eid in missing_ids
                if eid not in by_id
            ]
            raise KeyError(
                f"{len(missing)} requested event IDs were not found "
                f"in parquet store; examples={missing[:10]}"
            )

        for row_index, eid in enumerate(missing_ids):
            values = by_id[eid]

            for column_index, scale in enumerate(scales_to_fetch):
                vec = values[column_index]

                if vec is None:
                    raise KeyError(
                        f"scale {scale!r} was not computed for event_id={eid} "
                        f"(embedding is null in the store)"
                    )

                arr = np.asarray(vec, dtype=np.float32)

                if arr.shape != (dim,):
                    raise ValueError(
                        f"scale {scale!r} for event_id={eid} has shape "
                        f"{arr.shape}, expected ({dim},)"
                    )

                self._emb_cache[scale][eid] = arr

        for scale in scales_to_fetch:
            cache = self._emb_cache[scale]

            if len(cache) > self._emb_cache_max:
                cache.clear()

            for position in missing_positions_by_scale[scale]:
                eid = event_ids[position]
                outputs[scale][position] = cache[eid]

        return outputs


    def _fetch_scale_from_faiss(
        self, event_ids: Sequence[int], scale: str
    ) -> np.ndarray:
        assert self._index is not None
        dim = self._ensure_dim()
        n = len(event_ids)
        # Group by year for batched reconstruct_many
        positions = [self.get_pos(int(eid)) for eid in event_ids]
        years = self.pub_year[positions]

        out = np.empty((n, dim), dtype=np.float32)

        order = np.argsort(years, kind="stable")
        sorted_years = years[order]
        sorted_eids = np.asarray([int(event_ids[i]) for i in order], dtype=np.int64)

        start = 0
        n_ord = len(order)
        while start < n_ord:
            end = start + 1
            year = int(sorted_years[start])
            while end < n_ord and int(sorted_years[end]) == year:
                end += 1
            year_indices = self._index.get(year)
            if year_indices is None:
                raise KeyError(
                    f"No FAISS index for pub_year={year}. "
                    f"Available: {sorted(self._index.keys())}"
                )
            batch_ids = sorted_eids[start:end]
            out[order[start:end]] = year_indices[scale].reconstruct_many(batch_ids)
            start = end

        return out

    def _scale_for_ids(
        self, event_ids: Sequence[int], scale: str
    ) -> np.ndarray:
        if scale not in SCALES:
            raise ValueError(f"Unknown scale {scale!r}; must be one of {SCALES}")
        if self._index is not None:
            return self._fetch_scale_from_faiss(event_ids, scale)
        return self._fetch_scale_from_parquet(event_ids, scale)

    # --- selective single-scale access --------------------------------

    def get_scale_embedding(self, pos: int, scale: str) -> np.ndarray:
        eid = int(self.event_id[pos])
        return self._scale_for_ids([eid], scale)[0]

    def get_scale_embeddings(
        self,
        event_ids: Sequence[int],
        scale: str,
    ) -> np.ndarray:
        return self._scale_for_ids(event_ids, scale)

    # --- ensemble access -------------------------------------------------

    def get_ensemble_embedding(
        self,
        pos: int,
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        scales = _validate_scales(scales)
        if len(weights) != len(scales):
            raise ValueError(
                f"weights length {len(weights)} != scales length {len(scales)}"
            )
        eid = int(self.event_id[pos])
        out: Optional[np.ndarray] = None
        # TODO optimize
        for w, s in zip(weights, scales):
            vec = self._scale_for_ids([eid], s)[0]
            out = w * vec if out is None else out + w * vec
        return out.astype(np.float32)


    def get_embeddings(
        self,
        event_ids: Sequence[int],
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        scales = _validate_scales(scales)

        if len(weights) != len(scales):
            raise ValueError(
                f"weights length {len(weights)} != scales length {len(scales)}"
            )

        if not scales:
            return np.empty(
                (len(event_ids), self._ensure_dim()),
                dtype=np.float32,
            )

        matrices = self._fetch_scales_from_parquet(
            event_ids,
            scales,
        )

        out = np.zeros(
            (len(event_ids), self._ensure_dim()),
            dtype=np.float32,
        )

        for weight, scale in zip(weights, scales):
            out += weight * matrices[scale]

        return out


    def get_concatenated_embeddings(
        self,
        event_ids: Sequence[int],
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        scales = _validate_scales(scales)

        def _norm_rows(M: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return M / norms

        # TODO Optimize
        blocks = [_norm_rows(self._scale_for_ids(event_ids, s)) for s in scales]
        return np.concatenate(blocks, axis=1).astype(np.float32)


__all__ = [
    "write_observation_parquet",
    "ParquetObservationWriter",
    "ParquetObservationStream",
    "ParquetObservationLookup",
    "observation_schema",
]