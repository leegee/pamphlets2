#!/usr/bin/env python

# tier3/tier3_1_temporal_clusters.py

from __future__ import annotations

import argparse
import hashlib
import itertools
import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from lib.corpus_config import (
    CORPUS_TIER2_DB_PATH,
    EVENTSTORE_T1_PATH,
    TMP_DIR,
)

from lib.corpus_db import analysis_db_connection
from lib.concept_resolve import resolve_concepts
from lib.corpus_logging import logger
from lib.sqlite_vector_blob import vector_to_blob
from retrieval.models import SCALES
from lib.cluster import (
    LOCAL_UMAP_PARAMS,
    build_global_projection,
    leiden_cluster,
    project,
    compute_cluster_centroids,
)

from tier1.observation_store_api import (
    open_observation_lookup,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GLOBAL_PROJECTION_CACHE = (
    TMP_DIR / "tier3_global_projection.npz"
)

# Tier 3 clustering needs one embedding representation per event.
#
# The old implementation obtained vectors through the Zarr/FAISS lookup
# machinery and vector_id. The current observation lookup exposes embeddings
# directly by event_id and scale.
#
# Change this if inspection of the old load_vectors() implementation shows
# that a different scale was historically intended.
CLUSTER_SCALE = "medium"


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

YEAR_CLUSTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS concept_year_cluster_info (
    concept TEXT NOT NULL,
    pub_year INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    cluster_label TEXT,
    centroid_nx REAL,
    centroid_ny REAL,
    centroid_gnx REAL,
    centroid_gny REAL,
    centroid_vector BLOB,
    point_count INTEGER,
    description TEXT,
    relative_mass REAL,
    PRIMARY KEY (
        concept,
        pub_year,
        cluster_id
    )
);

CREATE INDEX IF NOT EXISTS idx_year_clusters
ON concept_year_cluster_info (
    concept,
    pub_year
);

-- Per-event cluster assignment for a (concept, pub_year) Leiden run.
--
-- process_concept_year() computes this mapping before collapsing events into
-- centroid rows. Downstream consumers need this table to answer:
--
--     "which concrete events belong to this node?"
--
-- events.cluster_id belongs to tier3_0's separate whole-concept clustering
-- pass and has an unrelated cluster ID space.
CREATE TABLE IF NOT EXISTS concept_year_event_cluster (
    concept TEXT NOT NULL,
    pub_year INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    PRIMARY KEY (
        concept,
        pub_year,
        event_id
    )
);

CREATE INDEX IF NOT EXISTS idx_year_event_cluster_lookup
ON concept_year_event_cluster (
    concept,
    pub_year,
    cluster_id
);

CREATE TABLE IF NOT EXISTS temporal_cluster_edges (
    concept TEXT,
    source_year INTEGER,
    source_cluster INTEGER,
    target_year INTEGER,
    target_cluster INTEGER,
    similarity REAL,
    edge_type TEXT,
    confidence REAL,
    PRIMARY KEY (
        concept,
        source_year,
        source_cluster,
        target_year,
        target_cluster,
        edge_type
    )
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_source
ON temporal_cluster_edges (
    concept,
    source_year,
    source_cluster
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_target
ON temporal_cluster_edges (
    concept,
    target_year,
    target_cluster
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_similarity
ON temporal_cluster_edges (
    concept,
    similarity
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_year_transition
ON temporal_cluster_edges (
    concept,
    source_year,
    target_year
);
"""


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def initialise_temporal_tables(con):
    """
    Ensure Tier 3.1 output tables exist.
    """
    con.executescript(
        YEAR_CLUSTER_SCHEMA
    )
    con.commit()


def clear_temporal_clusters(con):
    """
    Remove all Tier 3.1 output.

    Leaves Tier 2 source data untouched.
    """
    logger.info(
        "[tier3.1] clearing temporal cluster output"
    )

    con.execute(
        "DROP TABLE IF EXISTS concept_year_cluster_info"
    )

    con.execute(
        "DROP TABLE IF EXISTS concept_year_event_cluster"
    )

    con.execute(
        "DROP TABLE IF EXISTS temporal_cluster_edges"
    )

    con.commit()

    initialise_temporal_tables(
        con
    )


def delete_temporal_edges(
    con,
    concept,
):
    con.execute(
        """
        DELETE FROM temporal_cluster_edges
        WHERE concept=?
        """,
        (concept,),
    )


def delete_concept_clusters(
    con,
    concept,
):
    """
    Remove all Tier 3.1 cluster rows for a concept across every year.
    """

    con.execute(
        """
        DELETE FROM concept_year_cluster_info
        WHERE concept=?
        """,
        (concept,),
    )

    con.execute(
        """
        DELETE FROM concept_year_event_cluster
        WHERE concept=?
        """,
        (concept,),
    )


def sqlite_connection(
    path: Path,
    busy_timeout_ms: int = 30000,
):
    con = analysis_db_connection(path)
    con.execute( f"PRAGMA busy_timeout={busy_timeout_ms}" )
    con.execute( "PRAGMA journal_mode=WAL" )
    con.execute( "PRAGMA synchronous=NORMAL" )
    con.execute( "PRAGMA wal_autocheckpoint=1000" )
    con.execute( "PRAGMA locking_mode=NORMAL" )

    return con


def with_sqlite_retry(
    fn,
    retries=10,
    delay=0.5,
):
    for attempt in range(retries):
        try:
            return fn()

        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc):
                raise

            if attempt == retries - 1:
                raise

            wait = delay * (
                2 ** attempt
            )

            logger.warning(
                "[tier3.1] database locked, retry %d/%d after %.1fs",
                attempt + 1,
                retries,
                wait,
            )

            time.sleep(
                wait
            )


# ---------------------------------------------------------------------------
# Tier 2 event loading
# ---------------------------------------------------------------------------

def load_concept_event_rows(
    con,
    concept,
):
    """
    Load every Tier 2 event belonging to a concept.

    concept_field_events is the explicit concept-to-event association
    produced by Tier 2. The event table remains the source of event metadata.
    """

    rows = con.execute(
        """
        SELECT
            e.pub_year,
            e.event_id
        FROM concept_field_events f
        JOIN events e
            ON e.event_id=f.event_id
        WHERE f.concept=?
        ORDER BY
            e.pub_year,
            e.event_id
        """,
        (concept,),
    ).fetchall()

    if not rows:
        raise RuntimeError(
            f"[tier3.1] no Tier 2 events found for concept={concept!r}"
        )

    by_year = {}

    for pub_year, group in itertools.groupby(
        rows,
        key=lambda r: r[0],
    ):
        by_year[int(pub_year)] = [
            int(event_id)
            for _, event_id in group
        ]

    logger.info(
        "[tier3.1] %s: %d events across %d years",
        concept,
        len(rows),
        len(by_year),
    )

    return by_year


def load_event_vectors(
    lookup,
    event_ids,
    scale=CLUSTER_SCALE,
):
    """
    Retrieve Tier 1 embeddings directly by event_id.

    This replaces the old:

        ZarrEventLookup
            +
        yearly FAISS indexes
            +
        vector_id
            +
        load_vectors()

    path.

    The Tier 2 Lance pipeline already uses the same observation lookup API
    to retrieve scale-specific vectors by event ID.
    """

    if not event_ids:
        return (
            [],
            np.empty(
                (0, 0),
                dtype=np.float32,
            ),
        )

    vectors = lookup.get_scale_embeddings( event_ids, scale, )

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise RuntimeError( f"Expected 2-D embeddings for scale={scale}, got shape={vectors.shape}" )

    if len(vectors) != len(event_ids):
        raise RuntimeError( f"Embedding/event alignment mismatch: {len(event_ids)} event IDs but {len(vectors)} vectors" )

    return ( list(event_ids), vectors )


# ---------------------------------------------------------------------------
# SQLite writers
# ---------------------------------------------------------------------------

def write_year_cluster_info(
    con,
    concept,
    pub_year,
    cluster_records,
):
    rows = []

    for c in cluster_records:
        rows.append(
            (
                concept,
                pub_year,
                c["cluster_id"],
                (
                    "noise"
                    if c["cluster_id"] == -1
                    else None
                ),
                c["centroid_nx"],
                c["centroid_ny"],
                c["centroid_gnx"],
                c["centroid_gny"],
                vector_to_blob(
                    c["centroid_vector"]
                ),
                c["point_count"],
                c["relative_mass"],
                None,
            )
        )

    con.executemany(
        """
        INSERT INTO concept_year_cluster_info (
            concept,
            pub_year,
            cluster_id,
            cluster_label,
            centroid_nx,
            centroid_ny,
            centroid_gnx,
            centroid_gny,
            centroid_vector,
            point_count,
            relative_mass,
            description
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def write_year_event_cluster_map(
    con,
    concept,
    pub_year,
    event_ids,
    clusters,
):
    """
    Persist event_id -> cluster_id assignments.
    """

    rows = [
        (
            concept,
            pub_year,
            int(event_id),
            int(cluster_id),
        )
        for event_id, cluster_id
        in zip(
            event_ids,
            clusters,
        )
    ]

    con.executemany(
        """
        INSERT OR REPLACE INTO concept_year_event_cluster (
            concept,
            pub_year,
            event_id,
            cluster_id
        )
        VALUES (?,?,?,?)
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# Per-year clustering
# ---------------------------------------------------------------------------

def process_concept_year(
    con,
    lookup,
    concept,
    pub_year,
    event_ids,
    global_coords,
    resolution_parameter,
    n_neighbors,
):
    """
    Cluster one concept's events for one publication year.

    Invariant:
        event_ids, vectors, local coordinates, and cluster assignments
        remain in exactly the same order throughout this routine.
    """

    logger.info(
        "[tier3.1] %s %s: %d events",
        concept,
        pub_year,
        len(event_ids),
    )

    if not event_ids:
        return

    event_ids, vectors = load_event_vectors(
        lookup,
        event_ids,
        scale=CLUSTER_SCALE,
    )

    if len(event_ids) == 0:
        return

    local_coords = project(
        vectors,
        LOCAL_UMAP_PARAMS,
    )

    clusters = leiden_cluster(
        vectors,
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    # Global UMAP is deliberately disabled for this run.
    #
    # Local coordinates are temporarily reused for the global-coordinate
    # columns so downstream schema consumers can still operate. These
    # coordinates are not comparable between independent year projections.
    if global_coords is not None:

        global_xy = np.asarray(
            [
                global_coords[event_id]
                for event_id in event_ids
            ],
            dtype=np.float32,
        )

    else:

        global_xy = np.asarray(
            local_coords,
            dtype=np.float32,
        )

    cluster_records = compute_cluster_centroids(
        vectors,
        local_coords,
        global_xy,
        clusters,
    )

    total = sum(
        c["point_count"]
        for c in cluster_records
    )

    for cluster in cluster_records:

        cluster["relative_mass"] = (
            cluster["point_count"] / total
            if total > 0
            else 0.0
        )

    write_year_cluster_info(
        con,
        concept,
        pub_year,
        cluster_records,
    )

    write_year_event_cluster_map(
        con,
        concept,
        pub_year,
        event_ids,
        clusters,
    )


# ---------------------------------------------------------------------------
# Per-concept processing
# ---------------------------------------------------------------------------

def process_concept(
    con,
    lookup,
    concept,
    global_coords,
    resolution_parameter,
    n_neighbors,
):
    """
    Process every publication year for one concept.

    Existing Tier 3.1 output is removed only after the concept has been
    confirmed to have Tier 2 events.
    """

    by_year = load_concept_event_rows(
        con,
        concept,
    )

    delete_concept_clusters(
        con,
        concept,
    )

    for pub_year, event_ids in by_year.items():

        process_concept_year(
            con,
            lookup,
            concept,
            pub_year,
            event_ids,
            global_coords,
            resolution_parameter,
            n_neighbors,
        )

    con.commit()

# ---------------------------------------------------------------------------
# Per-concept processing
# ---------------------------------------------------------------------------

def process_concept(
    con,
    lookup,
    concept,
    global_coords,
    resolution_parameter,
    n_neighbors,
):
    """
    Process every year for one concept.
    """

    by_year = load_concept_event_rows( con, concept, )

    if not by_year:
        logger.warning( "[tier3.1] no events for concept=%s", concept, )
        return

    delete_concept_clusters( con, concept, )

    for pub_year, event_ids in by_year.items():
        process_concept_year(
            con,
            lookup,
            concept,
            pub_year,
            event_ids,
            global_coords,
            resolution_parameter,
            n_neighbors,
        )

    con.commit()


# ---------------------------------------------------------------------------
# Temporal edge construction
# ---------------------------------------------------------------------------

def load_year_clusters(
    con,
    concept,
    pub_year,
):
    rows = con.execute(
        """
        SELECT
            cluster_id,
            centroid_vector
        FROM concept_year_cluster_info
        WHERE
            concept=?
            AND pub_year=?
            AND cluster_id >= 0
        """,
        (
            concept,
            pub_year,
        ),
    ).fetchall()

    result = []

    for cluster_id, blob in rows:

        vector = np.frombuffer(
            blob,
            dtype=np.float32,
        ).copy()

        norm = np.linalg.norm(
            vector
        )

        if norm > 0:
            vector = vector / norm

        result.append(
            (
                int(cluster_id),
                vector,
            )
        )

    return result


def build_temporal_edges(
    con,
    concept,
    similarity_threshold=0.95,
):
    logger.info( "[tier3.1] building temporal edges %s", concept )

    delete_temporal_edges( con, concept )

    years = [
        row[0]
        for row in con.execute(
            """
            SELECT DISTINCT pub_year
            FROM concept_year_cluster_info
            WHERE
                concept=?
                AND cluster_id >= 0
            ORDER BY pub_year
            """,
            (concept,),
        )
    ]

    edges = []

    # Each year is read once and cached here. Consecutive iterations of
    # the loop below both need year Y (as target_year, then as the next
    # iteration's source_year); without this cache that meant a repeat
    # SQL round trip plus blob-decode/normalise for every interior year.
    year_clusters_cache = {}

    def get_year_clusters(year):
        cached = year_clusters_cache.get(year)
        if cached is None:
            cached = load_year_clusters(
                con,
                concept,
                year,
            )
            year_clusters_cache[year] = cached
        return cached

    for source_year, target_year in zip(
        years,
        years[1:],
    ):

        source_clusters = get_year_clusters(
            source_year,
        )

        target_clusters = get_year_clusters(
            target_year,
        )

        if not source_clusters or not target_clusters:
            continue

        source_ids = [
            x[0]
            for x in source_clusters
        ]

        source_vectors = np.vstack(
            [
                x[1]
                for x in source_clusters
            ]
        )

        target_ids = [
            x[0]
            for x in target_clusters
        ]

        target_vectors = np.vstack(
            [
                x[1]
                for x in target_clusters
            ]
        )

        similarity = cosine_similarity(
            source_vectors,
            target_vectors,
        )

        # ---------------------------------------------------------------
        # Continuations, and significant secondary relationships / splits
        # / merges, in a single pass. best_j (each source cluster's
        # strongest match in the target year) is needed by both: once to
        # decide the CONTINUATION edge, and again to exclude that same
        # pair from the SIGNIFICANT sweep. Previously each source cluster
        # ran np.argmax(row) twice, once per loop.
        # ---------------------------------------------------------------

        for i, source_cluster in enumerate(
            source_ids
        ):

            row = similarity[i]

            best_j = int(
                np.argmax(row)
            )

            best_score = float(
                row[best_j]
            )

            if len(row) > 1:

                sorted_scores = np.sort(
                    row
                )

                second_score = float(
                    sorted_scores[-2]
                )

            else:

                second_score = 0.0

            margin = (
                best_score
                - second_score
            )

            # Retained for diagnostic clarity.
            # Confidence intentionally remains on the raw similarity scale.
            _ = margin

            confidence = best_score

            if best_score >= similarity_threshold:

                edges.append(
                    (
                        concept,
                        source_year,
                        source_cluster,
                        target_year,
                        target_ids[best_j],
                        best_score,
                        "CONTINUATION",
                        confidence,
                    )
                )

            for j, target_cluster in enumerate(
                target_ids
            ):

                if j == best_j:
                    continue

                score = float(
                    row[j]
                )

                if score < similarity_threshold:
                    continue

                edges.append(
                    (
                        concept,
                        source_year,
                        source_cluster,
                        target_year,
                        target_cluster,
                        score,
                        "SIGNIFICANT",
                        score,
                    )
                )

    con.executemany(
        """
        INSERT OR REPLACE INTO temporal_cluster_edges (
            concept,
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            similarity,
            edge_type,
            confidence
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        edges,
    )

    logger.info( "[tier3.1] edges created: %d", len(edges) )


# ---------------------------------------------------------------------------
# Global projection cache
# ---------------------------------------------------------------------------

def event_id_hash(
    event_ids,
):
    h = hashlib.sha256()

    for event_id in event_ids:
        h.update(
            str(event_id).encode()
        )

    return h.hexdigest()


def load_or_build_global_projection(
    lookup,
    all_event_ids,
    cache_path,
):
    """
    Retained for later use.

    The caller currently has the expensive global UMAP build commented out.
    """

    if (
        cache_path is not None
        and cache_path.exists()
    ):

        logger.info( "[tier3.1] loading cached global projection from %s", cache_path )

        cached = np.load( cache_path, allow_pickle=False )

        cached_ids = cached[ "event_ids" ]
        cached_xy = cached[ "xy" ]
        cached_fingerprint = cached[ "fingerprint" ].item()

        if ( cached_fingerprint != event_id_hash(cached["event_ids"]) ):
            raise RuntimeError( "Global projection cache fingerprint mismatch" )

        if ( cached_fingerprint == event_id_hash(all_event_ids) ):
            return {
                int(event_id): xy
                for event_id, xy
                in zip(
                    cached_ids,
                    cached_xy,
                )
            }

        logger.info( "[tier3.1] cached global projection is stale (event set changed) -- rebuilding" )

    global_coords = build_global_projection( lookup, all_event_ids )

    if cache_path is not None:
        ids_arr = np.asarray(
            all_event_ids,
            dtype=np.int64,
        )

        xy_arr = np.asarray(
            [
                global_coords[event_id]
                for event_id in all_event_ids
            ],
            dtype=np.float32,
        )

        np.savez(
            cache_path,
            event_ids=ids_arr,
            xy=xy_arr,
            fingerprint=event_id_hash(
                all_event_ids
            ),
        )

        logger.info( "[tier3.1] cached global projection to %s", cache_path )

    return global_coords


# ---------------------------------------------------------------------------
# Parallel support
# ---------------------------------------------------------------------------

_WORKER_LOOKUP = None
_WORKER_GLOBAL_COORDS = None
_WORKER_CON = None


def _pin_single_threaded_math_libs():
    """
    Prevent process-level parallelism from being multiplied by internal
    OpenMP/BLAS/Numba worker pools.
    """

    os.environ[
        "OMP_NUM_THREADS"
    ] = "1"

    os.environ[
        "MKL_NUM_THREADS"
    ] = "1"

    os.environ[
        "OPENBLAS_NUM_THREADS"
    ] = "1"

    os.environ[
        "NUMEXPR_NUM_THREADS"
    ] = "1"

    try:
        import faiss

        faiss.omp_set_num_threads(
            1
        )

    except Exception:
        pass

    try:
        import numba

        numba.set_num_threads(
            1
        )

    except Exception:
        pass


def _init_worker(
    db_path,
    store_path,
    busy_timeout_ms,
):
    global _WORKER_CON, _WORKER_LOOKUP, _WORKER_GLOBAL_COORDS

    _pin_single_threaded_math_libs()

    _WORKER_CON = sqlite_connection(
        db_path,
        busy_timeout_ms=busy_timeout_ms,
    )

    _WORKER_LOOKUP = (
        open_observation_lookup(
            store_path
        )
    )

    # Global UMAP deliberately disabled.
    _WORKER_GLOBAL_COORDS = None


def _process_concept_worker(
    concept,
    similarity_threshold,
    resolution_parameter,
    n_neighbors,
):
    global _WORKER_LOOKUP, _WORKER_GLOBAL_COORDS, _WORKER_CON

    try:

        def write_concept():

            try:

                _WORKER_CON.execute(
                    "BEGIN IMMEDIATE"
                )

                process_concept(
                    _WORKER_CON,
                    _WORKER_LOOKUP,
                    concept,
                    _WORKER_GLOBAL_COORDS,
                    resolution_parameter,
                    n_neighbors,
                )

                build_temporal_edges(
                    _WORKER_CON,
                    concept,
                    similarity_threshold,
                )

                _WORKER_CON.commit()

            except Exception:

                if _WORKER_CON.in_transaction:

                    _WORKER_CON.rollback()

                raise

        with_sqlite_retry(
            write_concept
        )

        return (
            concept,
            None,
        )

    except Exception as exc:

        logger.exception( "[tier3.1] concept=%s failed in worker", concept )

        return (
            concept,
            repr(exc),
        )


def run_parallel(
    con,
    concepts,
    workers,
    db_path,
    store_path,
    similarity_threshold,
    resolution_parameter,
    n_neighbors,
):
    global _WORKER_LOOKUP, _WORKER_GLOBAL_COORDS, _WORKER_CON

    _WORKER_LOOKUP = None
    _WORKER_GLOBAL_COORDS = None
    _WORKER_CON = None

    # Workers create their own SQLite connections.
    con.close()

    ctx = mp.get_context(
        "fork"
        if "fork" in mp.get_all_start_methods()
        else "spawn"
    )

    failures = []

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(
            db_path,
            store_path,
            30000,
        ),
    ) as pool:

        from functools import partial

        worker = partial(
            _process_concept_worker,
            similarity_threshold=similarity_threshold,
            resolution_parameter=resolution_parameter,
            n_neighbors=n_neighbors,
        )

        for concept, err in pool.imap_unordered(
            worker,
            concepts,
        ):

            if err is None:
                logger.info( "[tier3.1] done: %s", concept )
            else:
                logger.error( "[tier3.1] FAILED: %s: %s", concept, err )
                failures.append( concept )

    if failures:
        raise SystemExit( f"[tier3.1] {len(failures)} concept(s) failed: {failures}" )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument( "-c", "--concept", default=None, )
    parser.add_argument( "-t", "--similarity-threshold", type=float, default=0.85, )
    parser.add_argument( "-r", "--resolution", type=float, default=0.8, help=( "Leiden resolution parameter (default: 0.8)" ), )
    parser.add_argument( "-n", "--neighbors", type=int, default=15, help=( "kNN graph neighbours (default: 15)" ), )

    # Retained for CLI compatibility.
    # The current direct event_id lookup path does not use FAISS indexes.
    parser.add_argument( "--mask", action="store_true", help=( "Retained for compatibility; currently unused by direct lookup" ), )
    parser.add_argument( "--clear", action="store_true", help=( "Delete all temporal cluster output before processing." ), )
    parser.add_argument( "--workers", type=int, default=1, help=( "Number of concepts to process in parallel" ), )

    args = parser.parse_args()

    logger.info( "[tier3.1] options: %s", vars(args) )
    logger.info( "[tier3.1] cluster embedding scale: %s", CLUSTER_SCALE )

    if CLUSTER_SCALE not in SCALES:
        raise RuntimeError( f"CLUSTER_SCALE={CLUSTER_SCALE!r} is not in available scales={SCALES!r}" )

    # -------------------------------------------------------------------
    # Tier 1 observation lookup
    # -------------------------------------------------------------------

    lookup = open_observation_lookup( EVENTSTORE_T1_PATH )

    # -------------------------------------------------------------------
    # Tier 2 SQLite database
    # -------------------------------------------------------------------

    con = sqlite_connection( CORPUS_TIER2_DB_PATH )
    initialise_temporal_tables( con )

    if args.clear:
        clear_temporal_clusters( con )

    # -------------------------------------------------------------------
    # Global projection
    # -------------------------------------------------------------------
    #
    # DELIBERATELY DISABLED.
    #
    # The call below can be extremely expensive because it builds the
    # global UMAP projection across all Tier 2 events.
    #
    # Uncomment when we want genuine corpus-wide/global coordinates again.
    # -------------------------------------------------------------------

    # all_rows = con.execute(
    #     """
    #     SELECT event_id
    #     FROM concept_field_events
    #     """
    # )
    #
    # all_event_ids = sorted(
    #     {
    #         int(row[0])
    #         for row in all_rows
    #     }
    # )
    #
    # global_coords = load_or_build_global_projection(
    #     lookup,
    #     all_event_ids,
    #     GLOBAL_PROJECTION_CACHE,
    # )

    global_coords = None

    logger.info( "[tier3.1] global UMAP projection disabled" )

    # -------------------------------------------------------------------
    # Concepts
    # -------------------------------------------------------------------

    concepts = [
        concept
        for concept, _
        in resolve_concepts(
            concept=args.concept
        )
    ]

    logger.info( "[tier3.1] processing %d concept(s)", len(concepts) )

    # -------------------------------------------------------------------
    # Parallel / sequential execution
    # -------------------------------------------------------------------

    if args.workers > 1:
        run_parallel(
            con,
            concepts,
            args.workers,
            CORPUS_TIER2_DB_PATH,
            EVENTSTORE_T1_PATH,
            args.similarity_threshold,
            args.resolution,
            args.neighbors,
        )

    else:

        for concept in concepts:
            process_concept(
                con,
                lookup,
                concept,
                global_coords,
                args.resolution,
                args.neighbors,
            )

            build_temporal_edges( con, concept, args.similarity_threshold, )

            # process_concept() commits its own writes, but
            # build_temporal_edges() does not. Without this commit, the
            # LAST concept's temporal edges are left in a pending
            # transaction when con.close() runs below; sqlite3 does not
            # commit on close(), so they were silently rolled back and
            # lost. Every other concept's edges happened to survive only
            # because the *next* concept's process_concept() commit
            # flushed them incidentally.
            con.commit()

        con.close()

    logger.info( "[tier3.1] Done." )


if __name__ == "__main__":
    mp.freeze_support()
    main()
