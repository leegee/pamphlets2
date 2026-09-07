# lib/corpus_db.py

"""
corpus_db.py - Corpus database access

Connections, schema (wip that should use .sql files), etc

Todo: rename stuff after including more than eebo

"""

import os
import psycopg
from psycopg import sql, Connection
import time

from lib.corpus_logging import logger
import lib.corpus_config as config

_DB_RETRIES = 3
_DB_RETRY_DELAY = 5  # seconds

dbname = os.environ.get("PGDATABASE", "eebo_parq")
# dbname = os.environ.get("CORPUS_PGDATABASE") or os.environ.get("PGDATABASE")
if not dbname:
    raise RuntimeError("Database name must be set via CORPUS_PGDATABASE or PGDATABASE")

host = os.environ.get("PGHOST", "localhost")
user = os.environ.get("PGUSER", "postgres")
password = os.environ.get("PGPASSWORD")
port = os.environ.get("PGPORT", 5432)

def get_connection(
    *,
    connect_timeout: int = 5,
    application_name: str = "corpus",
) -> Connection:
    """
    Create and return a PostgreSQL connection with autocommit disabled.
    Callers should use `with conn.transaction():` or call `conn.commit()`.
    Applies session-level tuning suitable for large bulk ingestion.
    """
    last_exc: Exception | None = None

    for attempt in range(1, _DB_RETRIES + 1):
        try:
            conn = psycopg.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                sslmode="disable",
                connect_timeout=connect_timeout,
                application_name=application_name,
            )
            conn.autocommit = False

            # Session-level tuning
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = OFF;")
                cur.execute("SET work_mem = '128MB';")
                cur.execute("SET maintenance_work_mem = '1GB';")
                cur.execute("SET temp_buffers = '32MB';")

            return conn

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"PostgreSQL connection attempt {attempt}/{_DB_RETRIES} failed: {exc}"
            )
            if attempt < _DB_RETRIES:
                time.sleep(_DB_RETRY_DELAY)

    raise RuntimeError("Could not establish PostgreSQL connection") from last_exc


def get_autocommit_connection(
    *,
    connect_timeout: int = 5,
    application_name: str = "borpus",
) -> Connection:
    """
    Get a fresh PostgreSQL connection in autocommit mode.
    Safe for COPY / bulk insert operations.
    Applies session-level tuning suitable for high-speed ingestion.
    """
    last_exc: Exception | None = None

    for attempt in range(1, _DB_RETRIES + 1):
        try:
            conn = psycopg.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                sslmode="disable",
                connect_timeout=connect_timeout,
                application_name=application_name,
                autocommit=True,  # enable immediately on connect
            )

            # Session-level tuning for bulk insert
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = OFF;")
                cur.execute("SET work_mem = '128MB';")
                cur.execute("SET maintenance_work_mem = '1GB';")
                cur.execute("SET temp_buffers = '32MB';")

            return conn

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"PostgreSQL autocommit connection attempt {attempt}/{_DB_RETRIES} failed: {exc}"
            )
            if attempt < _DB_RETRIES:
                time.sleep(_DB_RETRY_DELAY)

    raise RuntimeError(
        "Could not establish PostgreSQL autocommit connection"
    ) from last_exc



def init_db(conn: Connection, drop_existing: bool = True) -> None:
    """
    Initialise database schema.
    If drop_existing=True, all existing tables are dropped first.
    Intended for clean re-ingestion runs.
    """
    logger.info("[corpus_db] Initialising database schema")

    with conn.transaction():
        with conn.cursor() as cur:
            if drop_existing:
                logger.info("[corpus_db] Dropping existing tables")
                cur.execute("""
                    -- DROP MATERIALIZED VIEW IF EXISTS document_search CASCADE;
                    DROP MATERIALIZED VIEW IF EXISTS pamphlet_tokens CASCADE;
                    DROP MATERIALIZED VIEW IF EXISTS pamphlet_corpus CASCADE;

                    DROP TABLE IF EXISTS tokens CASCADE;
                    DROP TABLE IF EXISTS documents CASCADE;

                    DROP SEQUENCE IF EXISTS vector_id_seq;
                """)

            logger.info("[corpus_db] Creating tables")
            # Prototype assumption: doc_id is globally unique across loaded corpora (EEBO/ECCO).
            # Introducing a joint corpus/doc_id PK introduced too much work for consumers so
            # was rolled back and kept in git history.
            # Revisit if additional corpora introduce collisions.
            cur.execute("""
                /* Core document metadata */
                CREATE TABLE documents (
                    corpus TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    title TEXT,
                    author TEXT,
                    pub_year INTEGER,
                    publisher TEXT,
                    pub_place TEXT,
                    source_date_raw TEXT,
                    token_count INTEGER,
                    lang CHAR(3) NOT NULL DEFAULT 'eng',
                    PRIMARY KEY (doc_id)
                );

                CREATE SEQUENCE vector_id_seq;
                CREATE TABLE tokens (
                    corpus TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,
                    token TEXT NOT NULL,
                    raw_token text,
                    canonical TEXT,
                    vector_id BIGINT UNIQUE DEFAULT nextval('vector_id_seq'),
                    PRIMARY KEY ( doc_id, token_idx),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );
            """)

    logger.info("[corpus_db] Database schema initiated")



def drop_token_indexes(conn: Connection) -> None:
    logger.info("[corpus_db] Dropping token indexes")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                DROP INDEX IF EXISTS idx_tokens_token;
                DROP INDEX IF EXISTS idx_tokens_doc;
                DROP INDEX IF EXISTS idx_tokens_canonical;
                DROP INDEX IF EXISTS idx_tokens_token_lower;
                DROP INDEX IF EXISTS idx_documents_lang;
                DROP INDEX IF EXISTS idx_documents_filepath;
            """)
    logger.info("[corpus_db] Token indexes dropped")


def create_token_indexes(conn: Connection) -> None:
    logger.info("[corpus_db] Creating basic token indexes")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_token            ON tokens(token);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_doc              ON tokens(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_token_lower      ON tokens (lower(token));")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_lang          ON documents(lang);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_filepath      ON documents(filepath);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_canonical        ON tokens(canonical);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_filter        ON documents(lang, pub_year, token_count);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_corpus_doc_idx   ON tokens(corpus, doc_id, token_idx);")

    logger.info("[corpus_db] Basic token indexes created")


def create_views(conn: Connection) -> None:
    logger.info("[corpus_db] Creating view")

    earliest = config.CORPUS_MIN_YEAR
    latest   = config.CORPUS_MAX_YEAR

    # Create non-concurrent indexes and materialized views inside a transaction
    with conn.transaction():
        with conn.cursor() as cur:
            # Materialized view for pamphlet_corpus
            logger.info("[corpus_db] [corpus_db] Creating materialised view pamphlet_corpus")

            # Just in case I previously messed-up - TODO check and tidy.
            cur.execute("DROP MATERIALIZED VIEW IF EXISTS pamphlet_corpus CASCADE;")

            if config.FILTER_DOCUMENT_SIZE:
                size_filter = f"""
                    AND token_count BETWEEN {config.MIN_TOKENS_IN_DOC} AND {config.MAX_TOKENS_IN_DOC}
                """
            else:
                size_filter = ""

            logger.info(f"[corpus_db] CREATE MATERIALIZED VIEW pamphlet_corpus with dates {earliest} and {latest} and size filter of '{size_filter}'")
            cur.execute(f"""
                CREATE MATERIALIZED VIEW pamphlet_corpus AS
                SELECT *
                FROM documents
                WHERE pub_year >= {earliest}
                AND pub_year <= {latest}
                {size_filter}
                AND (lang = 'eng' OR lang IS NULL);
            """)

            # Index for fast joins (non-concurrent)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pamphlet_corpus_docid ON pamphlet_corpus(corpus, doc_id);")

            # Materialized view for pamphlet_tokens
            logger.info("[corpus_db] Creating materialised view pamphlet_tokens")
            cur.execute("DROP MATERIALIZED VIEW IF EXISTS pamphlet_tokens CASCADE;")

            # TODO Drop vector_id, no longer used
            cur.execute("""
                CREATE MATERIALIZED VIEW pamphlet_tokens AS
                SELECT
                    (t.doc_id || '_' || t.token_idx) AS token_occurrence_id,
                    t.corpus,
                    t.doc_id,
                    t.token_idx,
                    t.vector_id,
                    d.pub_year,
                    t.token,
                    t.canonical
                FROM tokens t
                JOIN pamphlet_corpus d ON t.doc_id = d.doc_id;
            """)

            # Materialized view for document_search
            # logger.info("[corpus_db] Creating materialised view document_search")
            # cur.execute("""
            #     CREATE MATERIALIZED VIEW IF NOT EXISTS document_search AS
            #     WITH numbered_tokens AS (
            #         SELECT
            #             t.corpus,
            #             t.doc_id,
            #             t.token,
            #             t.token_idx,
            #             (row_number() OVER (PARTITION BY t.corpus, t.doc_id ORDER BY t.token_idx) - 1) / 50000 AS block_idx
            #         FROM tokens t
            #         JOIN pamphlet_corpus pc ON pc.doc_id = t.doc_id
            #     ),
            #     block_text AS (
            #         SELECT
            #             corpus,
            #             doc_id,
            #             block_idx,
            #             string_agg(token, ' ') AS text
            #         FROM numbered_tokens
            #         GROUP BY corpus, doc_id, block_idx
            #     )
            #     SELECT
            #         d.corpus,
            #         d.doc_id,
            #         d.title,
            #         d.author,
            #         d.pub_year,
            #         d.pub_place,
            #         d.publisher,
            #         bt.block_idx,
            #         bt.text,
            #         to_tsvector('english', bt.text) AS tsv
            #     FROM pamphlet_corpus d
            #     JOIN block_text bt ON bt.doc_id = d.doc_id;
            # """)


def create_tiered_token_indexes(conn: Connection) -> None:
    # logger.info("[corpus_db] Creating tiered token indexes")
    # earliest = config.CORPUS_MIN_YEAR
    # latest   = config.CORPUS_MAX_YEAR
    # Create non-concurrent indexes and materialized views inside a transaction
    # with conn.transaction():
    #     with conn.cursor() as cur:
    #         # GIN index can stay in transaction
    #         cur.execute("CREATE INDEX IF NOT EXISTS idx_document_search_tsv ON document_search USING GIN(tsv);")
    #         cur.execute("CREATE INDEX IF NOT EXISTS idx_document_search_docid ON document_search(corpus, doc_id);")
    # logger.info("[corpus_db] create_tiered_token_indexes complete")
    logger.info("[corpus_db] create_tiered_token_indexes = noop")


def create_concurrent_indexes():
    logger.info("[corpus_db] create_concurrent_indexes enter to create CONCURRENT indexes")
    with get_autocommit_connection() as conn:
        with conn.cursor() as cur:
            with conn.cursor() as cur:
                cur.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_token_occurrence_id ON pamphlet_tokens(token_occurrence_id);")
                cur.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pt_doc_token_idx ON pamphlet_tokens(corpus, doc_id, token, token_idx);")
    logger.info("[corpus_db] create_concurrent_indexes complete")



def drop_tokens_fk(conn: Connection) -> None:
    logger.info("[corpus_db] Dropping tokens.doc_id foreign key")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE IF EXISTS tokens DROP CONSTRAINT IF EXISTS tokens_doc_id_fkey;")
    logger.info("[corpus_db] tokens.doc_id foreign key dropped")


def create_tokens_fk(conn: Connection) -> None:
    logger.info("[corpus_db] NOT creating tokens.doc_id foreign key as v slow and immutable data  makes it irrelevant")


def refresh_views(conn: Connection) -> None:
    logger.info("[corpus_db] Refreshing materialized views")

    with conn.transaction():
        with conn.cursor() as cur:
            for view in [
                "pamphlet_tokens", "pamphlet_corpus",
                # "document_search"
            ]:
                logger.info(f"[corpus_db] Refreshing {view}")
                cur.execute( sql.SQL("REFRESH MATERIALIZED VIEW {view}").format( view=sql.Identifier(view) ) )
    logger.info("[corpus_db] All views refreshed and committed")



def analysis_db_connection(db_path):
    import sqlite3
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


