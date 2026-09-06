#!/usr/bin/env python
"""
eebo_parse_tei.py - Multi-process streaming EEBO TEI XML ingestion pipeline

NB Corpus roots are defined in config.CORPUS_INPUT_DIRS

"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional
import argparse
import csv
import io
import os
import re
import sys
import tempfile
import traceback
import unicodedata

from psycopg import sql
import xml.etree.ElementTree as etree
import langdetect

import lib.corpus_config as config
import lib.corpus_db as corpus_db
import lib.eebo_ocr_fixes as eebo_ocr_fixes
from lib.corpus_logging import logger
from lib.wordlist_whiteness import WHITENESS, WHITENESS_WEIGHTS, WEAK_WHITENESS_PATTERNS

MIN_WHITENESS_SCORE = 2.0

NUM_WORKERS = 4
BATCH_DOCS = 100
BATCH_TOKENS = 10000

LOG_EVERY_N_DOCS = 100
ALLOWED_PUNCT = r"\.\,\;\:\!\?\'\"\-\(\)"

# Per-worker lazy cache
_ECCO_HEADER_INDEX = None

# Threading these through everything is silly.
MAX_DOCS: Optional[int] = None
SKIP_EXISTING_DOCS  = True

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def whiteness_seed_score(
    tokens: list[str],
) -> tuple[int, Counter[str]]:

    hits = Counter(
        token.lower()
        for token in tokens
        if token.lower() in WHITENESS
    )

    score = sum(
        count * WHITENESS_WEIGHTS.get(lemma, 1)
        for lemma, count in hits.items()
    )

    return score, hits


def passes_whiteness_filter( doc_id: str, tokens: list[str] ) -> bool:
    score, hits = whiteness_seed_score(tokens)

    if not hits:
        return False

    seeds = set(hits)

    # White/grey/gray alone are too non-diagnostic,
    # regardless of how frequently they occur.
    if any(seeds <= weak for weak in WEAK_WHITENESS_PATTERNS):
        return False

    if score < MIN_WHITENESS_SCORE:
        return False

    logger.info( f"[tier0] Whiteness filter PASS: {doc_id} score={score}, hits={dict(hits)}" )
    return True


def normalize_early_modern(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)
    text = text.replace("ſ", "s")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"-\s*", " ", text)
    # I'm really not sure we ought to be messing with actual alphabetic chars
    # text = re.sub(r"\bv(?=[aeiou])", "u", text)
    # text = re.sub(r"\bj(?=[aeiou])", "i", text)
    text = re.sub(r"tv\b", "ty", text)

    text = re.sub(rf"[^{ALLOWED_PUNCT}a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def render_text(node):
    """Render EEBO XML into plain text while preserving editorial GAP markers."""
    parts = []

    if node.text:
        parts.append(node.text)

    for child in node:
        local_name = child.tag.rsplit("}", 1)[-1]

        if local_name == "gap":
            extent = child.attrib.get("extent", "")
            m = re.search(r"(\d+)", extent)
            n = int(m.group(1)) if m else 1
            parts.append("_" * n)
        else:
            parts.append(render_text(child))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def extract_year(date_raw: str | None):
    if not date_raw:
        return None

    m = re.search(r"\b(\d{4})\b", date_raw)

    if not m:
        return None

    return int(m.group(1))


def year_in_corpus(pub_year: int | None):
    if pub_year is None:
        return False

    return (
        config.CORPUS_MIN_YEAR
        <= pub_year
        <= config.CORPUS_MAX_YEAR
    )


def safe_text(x):
    return x.text.strip() if x is not None and x.text else None


def to_doc_row(meta: dict) -> tuple:
    return (
        meta["corpus"],
        meta["doc_id"],
        meta["title"],
        meta["author"],
        meta["pub_year"],
        meta["publisher"],
        meta["pub_place"],
        meta["source_date_raw"],
        meta["token_count"],
        meta.get("filepath"),
        meta.get("lang"),
    )


def extract_language(tree, raw_text):
    lang_elem = tree.find(
        ".//tei:profileDesc//tei:language",
        TEI_NS,
    )

    lang = None

    if lang_elem is not None:
        lang = (
            lang_elem.attrib.get("ident")
            or safe_text(lang_elem)
        )

    if not lang:
        text_node = tree.find(".//tei:text", TEI_NS)
        if text_node is not None:
            lang = (
                text_node.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                or text_node.attrib.get("lang")
            )

    if lang:
        lang = lang.lower()

        if lang in {"eng", "en"}:
            return "eng"

        langs = re.findall(r"[a-z]{2,3}", lang)
        if langs:
            return langs[0][:3]

    try:
        detected = langdetect.detect(raw_text[:5000])
        return detected[:3] if detected else None
    except Exception:
        return None


def get_ecco_header_index():
    """
    Lazy-load ECCO header lookup on first ECCO document.
    Each worker gets its own cache.
    """
    global _ECCO_HEADER_INDEX
    if _ECCO_HEADER_INDEX is None:
        logger.info( f"[tier0 worker {os.getpid()}] loading ECCO header index" )
        _ECCO_HEADER_INDEX = {
            p.name.removesuffix(".hdr"): p
            for p in config.ECCO_HEADER_DIR.rglob("*.hdr")
        }
        logger.info( f"[tier0 worker {os.getpid()}] loaded {len(_ECCO_HEADER_INDEX)} ECCO headers" )
    return _ECCO_HEADER_INDEX


def find_ecco_header(doc_id):
    index = get_ecco_header_index()
    return index.get(doc_id)


def extract_ecco_header_metadata(header_path):
    tree = etree.parse(str(header_path))
    title = tree.findtext(".//TITLESTMT/TITLE")
    author = tree.findtext(".//TITLESTMT/AUTHOR")
    pub = tree.find(".//SOURCEDESC//PUBLICATIONSTMT")
    publisher = None
    pub_place = None
    date_raw = None
    if pub is not None:
        publisher = pub.findtext("PUBLISHER")
        pub_place = pub.findtext("PUBPLACE")
        date_raw = pub.findtext("DATE")
    year = None
    if date_raw:
        m = re.search(r"\b(\d{4})\b", date_raw)
        if m:
            year = int(m.group(1))

    return {
        "title": title,
        "author": author,
        "publisher": publisher,
        "pub_place": pub_place,
        "source_date_raw": date_raw,
        "date_raw": date_raw,
        "pub_year": year,
    }


def process_ecco_file(tree, xml_path):
    idg = tree.find(".//EEBO/IDG")

    if idg is None:
        logger.warning(f"[tier0] No IDG in {xml_path}")
        return None

    doc_id = idg.attrib["ID"]

    header_path = find_ecco_header(doc_id)
    if header_path is None:
        logger.warning(f"[tier0] No ECCO header for {doc_id} at {xml_path}")
        return None

    metadata = extract_ecco_header_metadata(header_path)

    pub_year = metadata["pub_year"]

    if pub_year is None:
        logger.warning(f"[tier0] No pub_year in {doc_id} {xml_path}")
        return None

    if not year_in_corpus(pub_year):
        return None

    body = tree.findall(".//TEXT/BODY")

    if not body:
        logger.warning(f"[tier0] No BODY for {doc_id} {xml_path}")
        return None

    raw_text = " ".join(
        render_text(b)
        for b in body
    )

    normalized = normalize_early_modern( eebo_ocr_fixes.apply_ocr_fixes(raw_text) )

    if len(normalized) < 100:
        return None

    tokens = re.findall( r"\w+|[^\w\s]", normalized )

    if not passes_whiteness_filter(doc_id, tokens):
        logger.debug( f"[tier0] Rejecting {doc_id}: not enough extreme-whiteness hits" )
        return None

    if len(tokens) > config.MAX_TOKENS_IN_DOC:
        logger.warning(
            f"[tier0 worker {os.getpid()}] "
            f"ECCO document {doc_id} has {len(tokens)} which exceeds the limit of MAX_TOKENS_IN_DOC {config.MAX_TOKENS_IN_DOC}"
        )
        # return None

    lang = extract_language(tree, raw_text)

    meta = {
        "doc_id": doc_id,
        "title": metadata["title"],
        "author": metadata["author"],
        "publisher": metadata["publisher"],
        "pub_place": metadata["pub_place"],
        "pub_year": pub_year,
        "source_date_raw": metadata["date_raw"],
        "token_count": len(tokens),
        "filepath": str( xml_path.relative_to(config.XML_ROOT_DIR).as_posix() ),
        "lang": lang,
    }

    return meta, tokens


def process_eebo_file(tree, xml_path):
    doc_id_elem = tree.find(".//tei:idno[@type='DLPS']", TEI_NS)

    if doc_id_elem is None or not doc_id_elem.text:
        logger.warning(
            f"[tier0] process_eebo_file bailing as doc_id_elem not found in {xml_path}"
        )
        return None

    doc_id = doc_id_elem.text.strip()

    title_elem = tree.find(".//tei:teiHeader//tei:titleStmt/tei:title", TEI_NS)
    author_elem = tree.find(".//tei:teiHeader//tei:titleStmt/tei:author", TEI_NS)
    pub_elem = tree.find(
        ".//tei:teiHeader//tei:sourceDesc//tei:publisher",
        TEI_NS,
    )
    place_elem = tree.find(
        ".//tei:teiHeader//tei:sourceDesc//tei:pubPlace",
        TEI_NS,
    )
    date_elem = tree.find(
        ".//tei:teiHeader//tei:sourceDesc//tei:date",
        TEI_NS,
    )

    date_raw = safe_text(date_elem)
    pub_year = extract_year(date_raw)

    if pub_year is None:
        logger.warning(
            f"[tier0 worker {os.getpid()}] No pub_year in {doc_id} at {xml_path}"
        )
        return None

    if not year_in_corpus(pub_year):
        return None

    body = tree.findall(".//tei:text/tei:body", TEI_NS)

    if not body:
        logger.warning(
            f"[tier0 worker {os.getpid()} ]"
            f"process_eebo_file bailing as BODY not defined in {xml_path}"
        )
        return None

    raw_text = " ".join(render_text(b) for b in body)

    normalized = normalize_early_modern(
        eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    )

    if len(normalized) < 100:
        logger.warning(
            f"[tier0 worker {os.getpid()}] "
            f"process_eebo_file bailing as normalised text length < 100 "
            f"in {xml_path}"
        )
        return None

    lang = extract_language(tree, raw_text)

    tokens = re.findall(r"\w+|[^\w\s]", normalized)

    if len(tokens) > config.MAX_TOKENS_IN_DOC:
        logger.warning(
            f"[tier0 worker {os.getpid()}] "
            f"EEBO document {doc_id} has {len(tokens)} tokens, "
            f"which exceeds MAX_TOKENS_IN_DOC "
            f"{config.MAX_TOKENS_IN_DOC}"
        )
        # return None

    meta = {
        "doc_id": doc_id,
        "title": safe_text(title_elem),
        "author": safe_text(author_elem),
        "publisher": safe_text(pub_elem),
        "pub_place": safe_text(place_elem),
        "pub_year": pub_year,
        "source_date_raw": date_raw,
        "token_count": len(tokens),
        "filepath": str(
            xml_path.relative_to(config.XML_ROOT_DIR).as_posix()
        ),
        "lang": lang,
    }

    return meta, tokens


def process_file(xml_path: Path, corpus):
    try:
        tree = etree.parse(str(xml_path))
    except Exception:
        logger.warning(f"[tier0] Failed to parse {xml_path}")
        return None

    if corpus == "eebo":
        return process_eebo_file(tree, xml_path)
    elif corpus == "ecco":
        return process_ecco_file(tree, xml_path)
    else:
        logger.warning(f"[tier0] Unknown corpus {corpus} - ignoring path {xml_path}")
        return None


def process_file_to_temp(xml_path: Path, corpus: str):
    result = process_file(xml_path, corpus)
    if not result:
        return None

    meta, tokens = result

    meta["corpus"] = corpus

    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", suffix=".tsv")
    writer = csv.writer(tmp, delimiter="\t")

    for i, tok in enumerate(tokens):
        writer.writerow([
            meta["corpus"],
            meta["doc_id"],
            i,
            tok
        ])

    tmp.close()

    return meta, tmp.name, len(tokens)


def stream_copy(table: str, columns: list[str], rows):
    if not rows:
        return

    stmt = sql.SQL(
        "COPY {table} ({fields}) FROM STDIN WITH (FORMAT text, DELIMITER E'\t', NULL '\\N')"
    ).format(
        table=sql.Identifier(table),
        fields=sql.SQL(', ').join(sql.Identifier(c) for c in columns)
    )

    def encode(row):
        return "\t".join(
            "\\N" if v is None else str(v).replace("\t", " ").replace("\n", " ")
            for v in row
        ) + "\n"

    buf = io.StringIO()
    for r in rows:
        buf.write(encode(r))
    buf.seek(0)

    with corpus_db.get_autocommit_connection() as conn:
        with conn.cursor() as cur:
            with cur.copy(stmt) as copy:
                copy.write(buf.read())


# Temporary solution:
def filter_existing_docs(rows, corpus):
    if not rows:
        return []

    doc_ids = [r[1] for r in rows]

    with corpus_db.get_connection() as conn:
        cur = conn.execute(
            """
            SELECT doc_id
            FROM documents
            WHERE corpus = %s
            AND doc_id = ANY(%s)
            """,
            (corpus, doc_ids,),
        )

        existing = {r[0] for r in cur.fetchall()}

    return [
        row for row in rows
        if row[1] not in existing
    ]


def _worker_ingest(files, batch_docs, batch_tokens, skip_existing_docs, corpus):
    logger.info( f"[tier0 worker {os.getpid()}] received {len(files)} {corpus} files" )
    doc_batch = []
    token_batch = []
    inserted_doc_ids = set()

    docs_seen = 0

    def log_progress():
        if docs_seen % LOG_EVERY_N_DOCS == 0 and docs_seen > 0:
            logger.info(f"[tier0 worker {os.getpid()}] ingested {docs_seen} docs")

    def flush_docs():
        nonlocal doc_batch, inserted_doc_ids
        if not doc_batch:
            return

        rows = doc_batch
        doc_batch = []

        if skip_existing_docs:
            rows = filter_existing_docs(rows, corpus)

        if not rows:
            return

        inserted_doc_ids.update(
            (r[0], r[1])
            for r in rows
        )

        stream_copy(
            "documents",
            [
                "corpus", "doc_id", "title", "author", "pub_year",
                "publisher", "pub_place", "source_date_raw",
                "token_count", "filepath", "lang",
            ],
            rows,
        )

    def flush_tokens():
        nonlocal token_batch
        if not token_batch:
            return

        rows = token_batch
        token_batch = []

        stream_copy(
            "tokens",
            ["corpus", "doc_id", "token_idx", "token"],
            rows,
        )

    for fp in files:
        try:
            result = process_file_to_temp(fp, corpus)
            if not result:
                continue

            meta, token_file, _ = result

            doc_batch.append(to_doc_row(meta))

            with open(token_file, "r", encoding="utf-8") as f:
                for line in f:
                    corpus, doc_id, idx, tok = line.rstrip("\n").split("\t")
                    token_batch.append((corpus, doc_id, int(idx), tok))

            if (
                len(doc_batch) >= batch_docs
                or len(token_batch) >= batch_tokens
            ):
                flush_docs()
                flush_tokens()

            docs_seen += 1
            log_progress()

        except Exception:
            logger.error(f"[tier0] FAILED FILE: {fp}")
            logger.error(traceback.format_exc())

    flush_docs()
    flush_tokens()
    logger.info(f"[tier0 worker {os.getpid()}] finished: {docs_seen} docs processed")


def ingest_xml_parallel(
    xml_dir: Path | None = None,
    max_workers: int     = 4,
    batch_docs: int      = 50,
    batch_tokens: int    = 50000,
    corpus: str          = None
):
    xml_files = list(xml_dir.rglob("*.xml"))

    logger.info(f"[tier0] Input directory: {xml_dir}")
    logger.info(f"[tier0] Found {len(xml_files)} XML files")

    for x in xml_files[:5]:
        logger.info(f"[tier0] Example: {x}")

    if MAX_DOCS is not None:
        xml_files = xml_files[:MAX_DOCS]

    chunks = [xml_files[i::max_workers] for i in range(max_workers)]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_worker_ingest, chunk, batch_docs, batch_tokens, SKIP_EXISTING_DOCS, corpus )
            for chunk in chunks
        ]
        for f in futures:
            f.result()


def validate_corpus_years():
    if not isinstance(config.CORPUS_MIN_YEAR, int):
        raise TypeError("CORPUS_MIN_YEAR must be an int")
    if not isinstance(config.CORPUS_MAX_YEAR, int):
        raise TypeError("CORPUS_MAX_YEAR must be an int")
    if config.CORPUS_MIN_YEAR > config.CORPUS_MAX_YEAR:
        raise ValueError(
            "CORPUS_MIN_YEAR cannot be greater than CORPUS_MAX_YEAR"
        )
    if config.CORPUS_MIN_YEAR < 1000:
        raise ValueError(f"Corpus CORPUS_MIN_YEAR appears invalid: '{config.CORPUS_MIN_YEAR}'")
    if config.CORPUS_MAX_YEAR > 2100:
        raise ValueError(f"Corpus CORPUS_MAX_YEAR appears invalid: '{config.CORPUS_MAX_YEAR}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to parse/add.")
    parser.add_argument("--create", action="store_true", help="Creates the database from scratch, deleting existing data. Requires manual confirmation.")
    parser.add_argument("--justindex", action="store_true", help="Does not parse any files but recreates database view and indicies.")
    args = parser.parse_args()

    validate_corpus_years() # Eventually allow flags

    global MAX_DOCS, SKIP_EXISTING_DOCS
    MAX_DOCS             = args.limit
    SKIP_EXISTING_DOCS   = not args.create

    with corpus_db.get_connection() as conn:
        if args.create:
            confirm = input("DESTROY DB? type YES: ")
            if confirm != "YES":
                sys.exit(1)
            corpus_db.init_db(conn)

        corpus_db.drop_token_indexes(conn)
        corpus_db.drop_tokens_fk(conn)
        conn.commit()

    if args.justindex:
        logger.info("[tier0] === Just indexing the DB, not parsing or ingesting files ===")
    else:
        for corpus, xml_dir in config.CORPUS_INPUT_DIRS.items():
            logger.info(f"[tier0] Process {corpus} from {xml_dir}")
            if not xml_dir.is_dir():
                parser.error(f"Input directory for {corpus} does not exist: {xml_dir}")

            ingest_xml_parallel(
                xml_dir=xml_dir,
                max_workers=NUM_WORKERS,
                batch_docs=BATCH_DOCS,
                batch_tokens=BATCH_TOKENS,
                corpus=corpus,
            )

    # Wait for other connections to finish
    with corpus_db.get_connection() as conn:
        while True:
            cur = conn.execute("""
                SELECT count(*) FROM pg_stat_activity
                WHERE datname = 'eebo'
                  AND pid <> pg_backend_pid()
                  AND state IN ('active', 'idle in transaction');
            """)
            n = cur.fetchone()[0]
            if n == 0:
                break

    with corpus_db.get_connection() as conn:
        corpus_db.create_tokens_fk(conn)
        corpus_db.create_token_indexes(conn)
        corpus_db.create_views(conn)
        corpus_db.create_tiered_token_indexes(conn)

    corpus_db.create_concurrent_indexes()


if __name__ == "__main__":
    main()
