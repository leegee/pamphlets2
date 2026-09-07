from __future__ import annotations

import argparse
import os
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np
import torch
import xxhash

from lib.corpus_config import CONCEPT_SETS, EMBED_BATCH_SIZE, LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import load_macberth
from lib.stopwords_min import STOPWORDS


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


WINDOW_CONFIGS = (
    {"name": "local", "size": 256, "stride": 128},
    {"name": "medium", "size": 512, "stride": 256},
    {"name": "broad", "size": 512, "stride": 384},
)

SCALES = tuple(config["name"] for config in WINDOW_CONFIGS)

LANCE_MODEL_NAME = "macberth"
LANCE_BUCKET_SIZE = 50


def stable_hash(key: str) -> int:
    # The same corpus/document/token occurrence must address the same
    # observation across all three Lance scale tables and across reruns.
    return xxhash.xxh64(key, seed=0).intdigest()


def normalise_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).strip().lower()


def seed_forms() -> set[str]:
    forms: set[str] = set()

    for rule in CONCEPT_SETS.values():
        forms.update(
            normalise_token(form)
            for form in rule["forms"]
        )

    return forms


def false_positive_forms() -> set[str]:
    forms: set[str] = set()

    for rule in CONCEPT_SETS.values():
        forms.update(
            normalise_token(form)
            for form in rule["false_positives"]
        )

    return forms


SEED_FORMS = seed_forms()
FALSE_POSITIVE_FORMS = false_positive_forms()


def is_seed(token: str) -> bool:
    value = normalise_token(token)

    return (
        value in SEED_FORMS
        and value not in FALSE_POSITIVE_FORMS
    )


def year_bucket(year: int) -> tuple[int, int]:
    start = (year // LANCE_BUCKET_SIZE) * LANCE_BUCKET_SIZE
    return start, start + LANCE_BUCKET_SIZE - 1


def lance_table_name(scale: str, year: int) -> str:
    start, end = year_bucket(year)

    return (
        f"{scale}__{LANCE_MODEL_NAME}__"
        f"{start:04d}_{end:04d}"
    )


@dataclass(slots=True)
class TokenRow:
    corpus: str
    doc_id: str
    token_idx: int
    token: str
    pub_year: int | None


@dataclass(slots=True)
class Observation:
    event_id: int
    corpus: str
    doc_id: str
    token: str
    token_idx: int
    pub_year: int | None


@dataclass(slots=True)
class EmbeddedObservation:
    observation: Observation
    vectors: dict[str, np.ndarray]


@dataclass(slots=True)
class DocBuffer:
    corpus: str
    doc_id: str
    pub_year: int | None
    rows: list[TokenRow]

    @property
    def tokens(self) -> list[str]:
        return [row.token for row in self.rows]

    def __bool__(self) -> bool:
        return bool(self.rows)


class MacBERThPipeline:
    def __init__(
        self,
        mac,
        *,
        batch_size: int = EMBED_BATCH_SIZE,
        mask_targets: bool = False,
    ) -> None:
        self.mac = mac
        self.tokenizer = mac.tokenizer
        self.model = mac.model
        self.device = mac.device
        self.batch_size = batch_size
        self.mask_targets = mask_targets

    def embed(
        self,
        document: DocBuffer,
        target_positions: set[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        if not target_positions:
            return {}

        results: dict[int, dict[str, np.ndarray]] = {
            position: {}
            for position in target_positions
        }

        encoded = self.tokenizer(
            document.tokens,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        word_ids = encoded.word_ids()

        if word_ids is None:
            raise RuntimeError(
                "MacBERTh tokenizer did not return word_ids; "
                "cannot align token occurrences to hidden states."
            )

        for config in WINDOW_CONFIGS:
            jobs = self._make_jobs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids,
                target_positions=target_positions,
                window_size=config["size"],
                stride=config["stride"],
            )

            for offset in range(0, len(jobs), self.batch_size):
                batch = jobs[offset : offset + self.batch_size]
                hidden = self._forward(batch)

                for job, vectors in zip(batch, hidden):
                    for word_position, vector in zip(
                        job["targets"],
                        vectors,
                    ):
                        results[word_position][
                            config["name"]
                        ] = vector

        incomplete = [
            position
            for position, vectors in results.items()
            if set(vectors) != set(SCALES)
        ]

        if incomplete:
            raise RuntimeError(
                f"{len(incomplete)} observations did not receive "
                "all three scale embeddings."
            )

        return results

    def _make_jobs(
        self,
        *,
        input_ids: list[int],
        attention_mask: list[int],
        word_ids: list[int | None],
        target_positions: set[int],
        window_size: int,
        stride: int,
    ) -> list[dict]:
        word_count = (
            max(
                word_id
                for word_id in word_ids
                if word_id is not None
            )
            + 1
        )

        jobs: list[dict] = []

        start_word = 0

        while start_word < word_count:
            encoded_positions = [
                index
                for index, word_id in enumerate(word_ids)
                if word_id is not None
                and start_word <= word_id < start_word + window_size
            ]

            if not encoded_positions:
                break

            encoded_start = min(encoded_positions)
            encoded_end = max(encoded_positions) + 1

            targets = [
                word_id
                for word_id in target_positions
                if start_word <= word_id < start_word + window_size
            ]

            if targets:
                window_ids = input_ids[
                    encoded_start:encoded_end
                ].copy()

                window_mask = attention_mask[
                    encoded_start:encoded_end
                ]

                target_encoded_positions = []

                for target in targets:
                    try:
                        relative = word_ids[
                            encoded_start:encoded_end
                        ].index(target)
                    except ValueError:
                        continue

                    target_encoded_positions.append(relative)

                    if self.mask_targets:
                        window_ids[relative] = (
                            self.tokenizer.mask_token_id
                        )

                if target_encoded_positions:
                    jobs.append(
                        {
                            "input_ids": window_ids,
                            "attention_mask": window_mask,
                            "targets": target_encoded_positions,
                        }
                    )

            if start_word + stride >= word_count:
                break

            start_word += stride

        return jobs

    def _forward(
        self,
        jobs: list[dict],
    ) -> list[list[np.ndarray]]:
        if not jobs:
            return []

        max_length = max(
            len(job["input_ids"])
            for job in jobs
        )

        input_ids = []
        attention_masks = []

        for job in jobs:
            padding = max_length - len(job["input_ids"])

            input_ids.append(
                job["input_ids"] + [0] * padding
            )

            attention_masks.append(
                job["attention_mask"] + [0] * padding
            )

        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )

        attention_tensor = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            output = self.model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                return_dict=True,
            )

        hidden = output.last_hidden_state.cpu().numpy()

        return [
            [
                hidden[
                    batch_index,
                    target_position,
                ].astype(np.float32, copy=False)
                for target_position in job["targets"]
            ]
            for batch_index, job in enumerate(jobs)
        ]


class EventWriter:
    def __init__(
        self,
        conn,
        lance_root: Path,
    ) -> None:
        self.conn = conn
        self.lance = lancedb.connect(str(lance_root))
        self.tables: dict[str, object] = {}

    def write(
        self,
        observations: list[EmbeddedObservation],
    ) -> int:
        if not observations:
            return 0

        self._write_postgres(observations)
        self._write_lance(observations)

        return len(observations)

    def _write_postgres(
        self,
        observations: list[EmbeddedObservation],
    ) -> None:
        rows = []

        for embedded in observations:
            observation = embedded.observation

            rows.append(
                (
                    observation.event_id,
                    observation.corpus,
                    observation.doc_id,
                    observation.token,
                    observation.token_idx,
                    observation.pub_year,
                )
            )

        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO events (
                    event_id,
                    corpus,
                    doc_id,
                    token,
                    token_idx,
                    pub_year
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                rows,
            )

        self.conn.commit()

    def _write_lance(
        self,
        observations: list[EmbeddedObservation],
    ) -> None:
        rows_by_table: dict[str, list[dict]] = defaultdict(list)

        for embedded in observations:
            observation = embedded.observation

            if observation.pub_year is None:
                raise ValueError(
                    f"Observation {observation.event_id} has no "
                    "publication year and cannot be assigned to "
                    "a chronological Lance table."
                )

            for scale in SCALES:
                table_name = lance_table_name(
                    scale,
                    observation.pub_year,
                )

                rows_by_table[table_name].append(
                    {
                        "event_id": observation.event_id,
                        "year": observation.pub_year,
                        "embedding_model": LANCE_MODEL_NAME,
                        "vector": embedded.vectors[scale].tolist(),
                    }
                )

        for table_name, rows in rows_by_table.items():
            table = self._open_table(table_name)
            table.add(rows, mode="append")

    def _open_table(self, table_name: str):
        if table_name in self.tables:
            return self.tables[table_name]

        if table_name not in self.lance.table_names():
            raise RuntimeError(
                f"Lance table does not exist: {table_name}. "
                "The Tier 1 producer never creates production "
                "tables implicitly."
            )

        table = self.lance.open_table(table_name)
        self.tables[table_name] = table

        return table


class CorpusProcessor:
    def __init__(
        self,
        conn,
        pipeline: MacBERThPipeline,
        writer: EventWriter,
        *,
        neighbour_radius: int = 256,
        report_every: int = 25,
    ) -> None:
        self.conn = conn
        self.pipeline = pipeline
        self.writer = writer
        self.neighbour_radius = neighbour_radius
        self.report_every = report_every

    def process(
        self,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
    ) -> None:
        documents = self._find_seed_documents(
            corpus=corpus,
            doc_id=doc_id,
        )

        logger.info(
            "[tier1] seed documents: %d",
            len(documents),
        )

        for number, (document_corpus, document_id) in enumerate(
            documents,
            start=1,
        ):
            started = time.perf_counter()

            document = self._load_document(
                document_corpus,
                document_id,
            )

            if document is None:
                continue

            seed_positions = {
                position
                for position, row in enumerate(document.rows)
                if is_seed(row.token)
            }

            if not seed_positions:
                continue

            target_positions = self._select_neighbours(
                seed_positions,
                len(document.rows),
            )

            embeddings = self.pipeline.embed(
                document,
                target_positions,
            )

            observations = []

            for position in sorted(target_positions):
                row = document.rows[position]

                observation = Observation(
                    event_id=stable_hash(
                        f"{row.corpus}:"
                        f"{row.doc_id}:"
                        f"{row.token_idx}"
                    ),
                    corpus=row.corpus,
                    doc_id=row.doc_id,
                    token=row.token,
                    token_idx=row.token_idx,
                    pub_year=row.pub_year,
                )

                observations.append(
                    EmbeddedObservation(
                        observation=observation,
                        vectors=embeddings[position],
                    )
                )

            written = self.writer.write(observations)

            elapsed = time.perf_counter() - started

            logger.info(
                "[tier1] %d/%d %s/%s: seeds=%d "
                "observations=%d written=%d elapsed=%.2fs",
                number,
                len(documents),
                document_corpus,
                document_id,
                len(seed_positions),
                len(observations),
                written,
                elapsed,
            )

            if number % self.report_every == 0:
                logger.info(
                    "[tier1] processed %d documents",
                    number,
                )

    def _find_seed_documents(
        self,
        *,
        corpus: str | None,
        doc_id: str | None,
    ) -> list[tuple[str, str]]:
        clauses = [
            "lower(t.token) = ANY(%s)",
        ]

        params: list[object] = [
            sorted(SEED_FORMS - FALSE_POSITIVE_FORMS),
        ]

        if corpus is not None:
            clauses.append("t.corpus = %s")
            params.append(corpus)

        if doc_id is not None:
            clauses.append("t.doc_id = %s")
            params.append(doc_id)

        sql = f"""
            SELECT DISTINCT
                t.corpus,
                t.doc_id
            FROM tokens AS t
            WHERE {" AND ".join(clauses)}
            ORDER BY t.corpus, t.doc_id
        """

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _load_document(
        self,
        corpus: str,
        doc_id: str,
    ) -> DocBuffer | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    t.corpus,
                    t.doc_id,
                    t.token_idx,
                    t.token,
                    d.pub_year
                FROM tokens AS t
                JOIN documents AS d
                  ON d.corpus = t.corpus
                 AND d.doc_id = t.doc_id
                WHERE t.corpus = %s
                  AND t.doc_id = %s
                ORDER BY t.token_idx
                """,
                (corpus, doc_id),
            )

            rows = cur.fetchall()

        if not rows:
            return None

        return DocBuffer(
            corpus=corpus,
            doc_id=doc_id,
            pub_year=rows[0][4],
            rows=[
                TokenRow(
                    corpus=row[0],
                    doc_id=row[1],
                    token_idx=row[2],
                    token=row[3],
                    pub_year=row[4],
                )
                for row in rows
            ],
        )

    def _select_neighbours(
        self,
        seed_positions: set[int],
        token_count: int,
    ) -> set[int]:
        positions: set[int] = set()

        for seed_position in seed_positions:
            start = max(
                0,
                seed_position - self.neighbour_radius,
            )

            end = min(
                token_count,
                seed_position + self.neighbour_radius + 1,
            )

            positions.update(
                range(start, end)
            )

        return positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Tier 1 observations directly from PostgreSQL "
            "through MacBERTh into Lance."
        )
    )

    parser.add_argument(
        "--corpus",
        default=None,
    )

    parser.add_argument(
        "--doc-id",
        default=None,
    )

    parser.add_argument(
        "--neighbour-radius",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Replace target tokens with [MASK] before embedding.",
    )

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=Path(LANCE_INDEXES_DIR),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_num_threads(
        int(os.environ.get("OMP_NUM_THREADS", "4"))
    )
    torch.set_num_interop_threads(1)

    conn = get_connection()

    try:
        mac = load_macberth()

        pipeline = MacBERThPipeline(
            mac,
            batch_size=args.batch_size,
            mask_targets=args.mask,
        )

        writer = EventWriter(
            conn,
            args.lance_root,
        )

        processor = CorpusProcessor(
            conn,
            pipeline,
            writer,
            neighbour_radius=args.neighbour_radius,
            report_every=args.report_every,
        )

        processor.process(
            corpus=args.corpus,
            doc_id=args.doc_id,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
