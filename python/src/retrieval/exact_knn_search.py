#!/usr/bin/env python
"""
exact_knn_search.py — Out-of-core exact kNN over Parquet observation shards

Implements unrestricted exact nearest-neighbour search when the full
embedding matrix does not fit in RAM:

    for each shard (Parquet part) in parallel:
        load shard vectors into RAM
        score all queries against the shard (exact IP on L2-normalised rows)
        keep per-query local top-k
    merge local heaps → global exact top-k

Shards are work units only. Every database vector is scored; nothing is
pruned by semantics. Parallelism reduces wall time, not total FLOPs.

Layout expected (hive-style, same as parquet_observation_backend):

    <root>/
        year=1640/part-*.parquet
        year=1641/part-*.parquet
        ...

Optional corpus= prefix directories are also discovered.

Usage
-----
    from exact_knn_search import discover_shards, exact_knn

    shards = discover_shards("/data/tier1_parquet")
    scores, ids = exact_knn(queries, shards, k=25, scale="medium", workers=4)

CLI
---
    python exact_knn_search.py \\
        --store /data/tier1_parquet \\
        --query-event-ids 1001,1002,1003 \\
        --k 25 --scale medium --workers 4
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence
from retrieval.models import SCALES
import numpy as np
import pyarrow.parquet as pq

ScaleName = Literal[SCALES]
DEFAULT_WEIGHTS = (0.25, 0.50, 0.25)

_YEAR_RE = re.compile(r"year=(\d+)$")
_CORPUS_RE = re.compile(r"corpus=([^/]+)$")


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shard:
    """One exact-search work unit = one Parquet part file."""

    path: Path
    corpus: Optional[str]
    year: Optional[int]
    n_rows: int

    @property
    def shard_id(self) -> str:
        return str(self.path)


def discover_shards(root: str | Path) -> list[Shard]:
    """
    Walk a Parquet observation store and return one Shard per part file.

    Supports:
        root/year=Y/part-*.parquet
        root/corpus=C/year=Y/part-*.parquet
        root/**/*.parquet  (year/corpus inferred from path when present)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"store root not found: {root}")

    shards: list[Shard] = []
    for path in sorted(root.rglob("*.parquet")):
        if path.name.startswith("."):
            continue
        corpus, year = _parse_partition(path, root)
        try:
            meta = pq.ParquetFile(path).metadata
            n_rows = int(meta.num_rows) if meta is not None else 0
        except Exception:
            n_rows = 0
        shards.append(Shard(path=path, corpus=corpus, year=year, n_rows=n_rows))
    return shards


def _parse_partition(path: Path, root: Path) -> tuple[Optional[str], Optional[int]]:
    corpus = None
    year = None
    for part in path.relative_to(root).parts[:-1]:
        m = _YEAR_RE.match(part)
        if m:
            year = int(m.group(1))
            continue
        m = _CORPUS_RE.match(part)
        if m:
            corpus = m.group(1)
    return corpus, year


def write_shard_manifest(shards: Sequence[Shard], dest: str | Path) -> None:
    """Optional CSV manifest for ops / scheduling."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        f.write("shard_id,corpus,year,n_rows,path\n")
        for s in shards:
            f.write(
                f"{s.shard_id},{s.corpus or ''},{s.year or ''},{s.n_rows},{s.path}\n"
            )


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def _list_col_to_matrix(col, dim: Optional[int] = None) -> np.ndarray:
    """Arrow fixed-size list column → (n, dim) float32."""
    import pyarrow as pa

    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        flat = col.values.to_numpy(zero_copy_only=False)
        n = len(col)
        d = col.type.list_size
        return np.asarray(flat, dtype=np.float32).reshape(n, d)
    arr = np.asarray(col.to_pylist(), dtype=np.float32)
    if arr.ndim == 1 and dim:
        return arr.reshape(-1, dim)
    return arr


def load_shard_vectors(
    path: str | Path,
    scale: ScaleName = "medium",
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (event_ids, vectors) for one Parquet part.

    vectors are L2-normalised float32 so inner product == cosine.
    scale='ensemble' uses weighted sum of local/medium/broad then renorm.
    """
    path = Path(path)
    table = pq.read_table(
        path,
        columns=["event_id", "emb_local", "emb_medium", "emb_broad"]
        if scale == "ensemble"
        else ["event_id", f"emb_{scale}"],
    )
    eids = table.column("event_id").to_numpy().astype(np.int64, copy=False)

    if scale == "ensemble":
        local = _list_col_to_matrix(table.column("emb_local"))
        medium = _list_col_to_matrix(table.column("emb_medium"))
        broad = _list_col_to_matrix(table.column("emb_broad"))
        vecs = (
            weights[0] * local + weights[1] * medium + weights[2] * broad
        ).astype(np.float32)
    else:
        vecs = _list_col_to_matrix(table.column(f"emb_{scale}"))

    vecs = _l2_normalize_rows(vecs)
    return eids, vecs


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


# ---------------------------------------------------------------------------
# Per-shard scoring (runs in worker processes)
# ---------------------------------------------------------------------------

def _score_shard(
    path: str,
    queries: np.ndarray,
    k: int,
    scale: str,
    weights: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact top-k of queries against one shard.

    Returns
    -------
    scores : (n_queries, k) float32  — descending, -inf padded if shard < k
    ids    : (n_queries, k) int64    — -1 padded
    """
    eids, vecs = load_shard_vectors(path, scale=scale, weights=weights)  # type: ignore[arg-type]
    if len(eids) == 0:
        nq = queries.shape[0]
        return (
            np.full((nq, k), -np.inf, dtype=np.float32),
            np.full((nq, k), -1, dtype=np.int64),
        )

    # (n_queries, n_shard) cosine == IP on normalised rows
    sims = queries @ vecs.T

    kk = min(k, sims.shape[1])
    # argpartition then sort the top-kk
    idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_scores = np.take_along_axis(sims, top_idx, axis=1)
    top_ids = eids[top_idx]

    if kk < k:
        nq = queries.shape[0]
        pad_s = np.full((nq, k - kk), -np.inf, dtype=np.float32)
        pad_i = np.full((nq, k - kk), -1, dtype=np.int64)
        top_scores = np.concatenate([top_scores, pad_s], axis=1)
        top_ids = np.concatenate([top_ids, pad_i], axis=1)

    return top_scores.astype(np.float32), top_ids.astype(np.int64)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_topk(
    score_blocks: Sequence[np.ndarray],
    id_blocks: Sequence[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge several (n_queries, k) results into a global (n_queries, k).

    Exact for a partition of the database when each block holds its true
    local top-k (or more).
    """
    if not score_blocks:
        raise ValueError("no blocks to merge")

    scores = np.concatenate(list(score_blocks), axis=1)
    ids = np.concatenate(list(id_blocks), axis=1)

    # Deduplicate event_ids per query keeping highest score
    nq, _ = scores.shape
    out_s = np.full((nq, k), -np.inf, dtype=np.float32)
    out_i = np.full((nq, k), -1, dtype=np.int64)

    for q in range(nq):
        sc = scores[q]
        idrow = ids[q]
        best: dict[int, float] = {}
        for s, i in zip(sc.tolist(), idrow.tolist()):
            i = int(i)
            if i < 0 or not np.isfinite(s):
                continue
            prev = best.get(i)
            if prev is None or s > prev:
                best[i] = float(s)
        if not best:
            continue
        items = sorted(best.items(), key=lambda t: -t[1])[:k]
        for j, (eid, s) in enumerate(items):
            out_i[q, j] = eid
            out_s[q, j] = s

    return out_s, out_i


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def exact_knn(
    queries: np.ndarray,
    shards: Sequence[Shard] | Sequence[Path | str],
    *,
    k: int = 25,
    scale: ScaleName = "medium",
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    workers: int = 1,
    pool: Literal["process", "thread"] = "process",
    exclude_self: bool = True,
    query_event_ids: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact unrestricted kNN of queries against all listed shards.

    Parameters
    ----------
    queries
        (n_queries, dim) float32. Will be L2-normalised in-place copy.
    shards
        Shard objects or paths to Parquet parts. All parts are scored.
    k
        Neighbours per query.
    scale
        Which embedding column(s) to use.
    workers
        Parallel shard scorers. 1 = sequential.
    pool
        'process' isolates BLAS better; 'thread' avoids pickle overhead
        on small shards.
    exclude_self
        If query_event_ids is provided, drop matches where neighbour id
        equals the query's event_id.
    query_event_ids
        Optional (n_queries,) ids aligned to query rows for self-filtering.

    Returns
    -------
    scores : (n_queries, k) float32
    ids    : (n_queries, k) int64   (-1 if fewer than k found)
    """
    queries = _l2_normalize_rows(np.asarray(queries, dtype=np.float32))
    if queries.ndim != 2:
        raise ValueError("queries must be (n_queries, dim)")

    paths: list[str] = []
    for s in shards:
        if isinstance(s, Shard):
            paths.append(str(s.path))
        else:
            paths.append(str(s))

    if not paths:
        nq = queries.shape[0]
        return (
            np.full((nq, k), -np.inf, dtype=np.float32),
            np.full((nq, k), -1, dtype=np.int64),
        )

    # Oversample when dropping self-matches so the final list still has k hits.
    need_self_filter = bool(exclude_self and query_event_ids is not None)
    k_search = k + 1 if need_self_filter else k

    w = tuple(float(x) for x in weights)
    score_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []

    if workers <= 1 or len(paths) == 1:
        for p in paths:
            sc, ids = _score_shard(p, queries, k_search, scale, w)
            score_blocks.append(sc)
            id_blocks.append(ids)
    else:
        Executor = ProcessPoolExecutor if pool == "process" else ThreadPoolExecutor
        with Executor(max_workers=workers) as ex:
            futs = {
                ex.submit(_score_shard, p, queries, k_search, scale, w): p
                for p in paths
            }
            for fut in as_completed(futs):
                sc, ids = fut.result()
                score_blocks.append(sc)
                id_blocks.append(ids)

    scores, ids = merge_topk(score_blocks, id_blocks, k_search)

    if need_self_filter:
        scores, ids = _exclude_self(scores, ids, query_event_ids, k)

    return scores, ids


def _exclude_self(
    scores: np.ndarray,
    ids: np.ndarray,
    query_event_ids: Sequence[int],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove self-matches and compact rows to the left."""
    qids = np.asarray(list(query_event_ids), dtype=np.int64)
    nq = scores.shape[0]
    out_s = np.full((nq, k), -np.inf, dtype=np.float32)
    out_i = np.full((nq, k), -1, dtype=np.int64)
    for q in range(nq):
        mask = ids[q] != qids[q]
        sc = scores[q, mask]
        idrow = ids[q, mask]
        take = min(k, len(sc))
        out_s[q, :take] = sc[:take]
        out_i[q, :take] = idrow[:take]
    return out_s, out_i


def exact_knn_from_store(
    store_root: str | Path,
    queries: np.ndarray,
    *,
    k: int = 25,
    scale: ScaleName = "medium",
    workers: int = 1,
    year_filter: Optional[Iterable[int]] = None,
    corpus_filter: Optional[Iterable[str]] = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    discover_shards(store_root) then exact_knn, with optional partition filters.

    Filters only limit which **files** are scanned (mechanical sharding),
    not which tokens/senses are eligible. Omitting filters = full unrestricted
    scan of the store.
    """
    shards = discover_shards(store_root)
    if year_filter is not None:
        years = set(int(y) for y in year_filter)
        shards = [s for s in shards if s.year is None or s.year in years]
    if corpus_filter is not None:
        corpora = set(corpus_filter)
        shards = [s for s in shards if s.corpus is None or s.corpus in corpora]
    return exact_knn(queries, shards, k=k, scale=scale, workers=workers, **kwargs)


def _filter_shards(
    shards: Sequence[Shard],
    *,
    year_filter: Optional[Iterable[int]] = None,
    corpus_filter: Optional[Iterable[str]] = None,
) -> list[Shard]:
    out = list(shards)
    if year_filter is not None:
        years = set(int(y) for y in year_filter)
        out = [s for s in out if s.year is None or s.year in years]
    if corpus_filter is not None:
        corpora = set(corpus_filter)
        out = [s for s in out if s.corpus is None or s.corpus in corpora]
    return out


def _query_vectors_for_positions(
    lookup,
    positions: np.ndarray,
    scale: ScaleName,
    store_root: Path,
    shards: Sequence[Shard],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (query_matrix, query_event_ids) for the given lookup positions.

    Prefer in-memory scale embeddings on the lookup when present (FAISS
    attach_index path). Otherwise load the vectors from the Parquet store
    by event_id so exact search needs no FAISS index at all.
    """
    positions = np.asarray(positions, dtype=np.int64)
    event_ids = np.asarray(lookup.event_id[positions], dtype=np.int64)

    emb_attr = f"emb_{scale}" if scale != "ensemble" else "emb_medium"
    emb = getattr(lookup, emb_attr, None)
    if emb is not None and scale != "ensemble":
        try:
            queries = np.asarray(emb[positions], dtype=np.float32)
            if queries.ndim == 2 and queries.shape[0] == len(positions):
                return queries, event_ids
        except Exception:
            pass

    # Fall back: scan shards for these event_ids (small sets relative to corpus).
    want = {int(e) for e in event_ids.tolist()}
    found: dict[int, np.ndarray] = {}
    for shard in shards:
        eids, vecs = load_shard_vectors(shard.path, scale=scale)
        for i, eid in enumerate(eids):
            eid = int(eid)
            if eid in want and eid not in found:
                found[eid] = vecs[i]
        if len(found) >= len(want):
            break
    missing = want - set(found)
    if missing:
        raise KeyError(
            f"query event_ids not found in store for scale={scale}: "
            f"{sorted(missing)[:20]}"
        )
    queries = np.stack([found[int(e)] for e in event_ids.tolist()], axis=0)
    return queries, event_ids


def multiscale_exact_search(
    store_root: str | Path,
    lookup,
    positions,
    top_n: int,
    *,
    pub_year: int | None = None,
    rrf_k: int = 60,
    oversample: int = 3,
    workers: int = 1,
    pool: Literal["process", "thread"] = "thread",
    shards: Optional[Sequence[Shard]] = None,
    corpus_filter: Optional[Iterable[str]] = None,
) -> list[list[dict]]:
    """
    Exact multi-scale kNN with the same return shape as
    ``eebo_faiss.multiscale_search``.

    Runs unrestricted (or year-filtered) exact search per scale against
    Parquet shards, then fuses local/medium/broad ranked lists with RRF.

    Parameters mirror FAISS multiscale_search so Tier 2 call sites can
    switch backends without changing downstream neighbour formatting.
    """
    store_root = Path(store_root)
    positions = np.asarray(positions, dtype=np.int64)
    n_queries = len(positions)
    search_k = top_n * oversample
    scales: tuple[ScaleName, ...] = SCALES

    all_shards = list(shards) if shards is not None else discover_shards(store_root)
    year_filter = [pub_year] if pub_year is not None else None
    active = _filter_shards(
        all_shards, year_filter=year_filter, corpus_filter=corpus_filter
    )
    if not active:
        return [[] for _ in range(n_queries)]

    # Query vectors: prefer lookup embeddings; else load from the *unfiltered*
    # shard list so seeds still resolve if they live in another partition.
    per_scale_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    query_event_ids: Optional[np.ndarray] = None
    for scale in scales:
        queries, qids = _query_vectors_for_positions(
            lookup, positions, scale, store_root, all_shards
        )
        query_event_ids = qids
        scores, ids = exact_knn(
            queries,
            active,
            k=search_k,
            scale=scale,
            workers=workers,
            pool=pool,
            exclude_self=True,
            query_event_ids=qids.tolist(),
        )
        per_scale_results[scale] = (scores, ids)

    assert query_event_ids is not None
    fused: list[list[dict]] = []
    for i in range(n_queries):
        scale_scores = {
            scale: {
                int(nid): float(score)
                for nid, score in zip(
                    per_scale_results[scale][1][i],
                    per_scale_results[scale][0][i],
                )
                if int(nid) != -1 and np.isfinite(score)
            }
            for scale in scales
        }
        ranked_lists = [list(scale_scores[scale].keys()) for scale in scales]
        fused_ids = _reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=top_n)
        fused.append(
            [
                {
                    "event_id": eid,
                    "rrf_score": rrf_score,
                    "score_local": scale_scores["local"].get(eid),
                    "score_medium": scale_scores["medium"].get(eid),
                    "score_broad": scale_scores["broad"].get(eid),
                }
                for eid, rrf_score in fused_ids
            ]
        )
    return fused


def _reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """RRF over ranked event_id lists (same formula as eebo_faiss)."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, eid in enumerate(ranked, start=1):
            if eid == -1:
                continue
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_n] if top_n else fused


# ---------------------------------------------------------------------------
# CLI (minimal: score random / listed event embeddings from the store)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exact out-of-core kNN over a Parquet observation store"
    )
    p.add_argument("--store", required=True, help="Parquet observation store root")
    p.add_argument(
        "--query-event-ids",
        type=str,
        default=None,
        help="Comma-separated event_ids to use as queries (loaded from store)",
    )
    p.add_argument("--k", type=int, default=25)
    p.add_argument(
        "--scale",
        choices=["local", "medium", "broad", "ensemble"],
        default="medium",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--pool",
        choices=["process", "thread"],
        default="thread",
        help="thread is safer for small tests; process for heavy BLAS isolation",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Write shard manifest CSV to this path",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Score only the first N shards (smoke tests)",
    )
    return p.parse_args()


def _load_query_vectors_by_ids(
    store_root: Path, event_ids: Sequence[int], scale: ScaleName
) -> np.ndarray:
    """Scan shards until all query ids are found (exact, small id sets)."""
    want = {int(e) for e in event_ids}
    found: dict[int, np.ndarray] = {}
    for shard in discover_shards(store_root):
        eids, vecs = load_shard_vectors(shard.path, scale=scale)
        for i, eid in enumerate(eids):
            eid = int(eid)
            if eid in want and eid not in found:
                found[eid] = vecs[i]
        if len(found) >= len(want):
            break
    missing = want - set(found)
    if missing:
        raise KeyError(f"query event_ids not in store: {sorted(missing)[:20]}")
    return np.stack([found[int(e)] for e in event_ids], axis=0)


def main():
    args = parse_args()
    root = Path(args.store)
    shards = discover_shards(root)
    print(f"shards={len(shards)} total_rows={sum(s.n_rows for s in shards):,}")
    if args.manifest:
        write_shard_manifest(shards, args.manifest)
        print(f"wrote manifest → {args.manifest}")

    if not args.query_event_ids:
        print("No --query-event-ids; manifest/discovery only.")
        return

    qids = [int(x) for x in args.query_event_ids.split(",") if x.strip()]
    queries = _load_query_vectors_by_ids(root, qids, args.scale)  # type: ignore[arg-type]
    if args.max_shards is not None:
        shards = shards[: args.max_shards]

    scores, ids = exact_knn(
        queries,
        shards,
        k=args.k,
        scale=args.scale,  # type: ignore[arg-type]
        workers=args.workers,
        pool=args.pool,  # type: ignore[arg-type]
        query_event_ids=qids,
    )
    for qi, qid in enumerate(qids):
        print(f"\nquery event_id={qid}")
        for j in range(args.k):
            if ids[qi, j] < 0:
                break
            print(f"  {j+1:2d}. id={ids[qi, j]}  score={scores[qi, j]:.6f}")


if __name__ == "__main__":
    main()
