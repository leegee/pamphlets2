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
from retrieval.models import SCALES

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


WINDOW_CONFIGS = (
    {"name": "local", "size": 256, "stride": 128},
    {"name": "medium", "size": 512, "stride": 256},
    {"name": "broad", "size": 512, "stride": 384},
)

LANCE_MODEL_NAME = "macberth"
LANCE_BUCKET_SIZE = 50


def stable_hash(key: str) -> int:
    # PostgreSQL BIGINT is signed, whereas xxhash returns an unsigned
    # 64-bit integer. Keep the stable lower 63 bits so the same occurrence
    # receives the same PostgreSQL-safe ID on every rerun.
    return xxhash.xxh64(key, seed=0).intdigest() & 0x7FFFFFFFFFFFFFFF


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
                    for target, vector in zip(
                        job["targets"],
                        vectors,
                    ):
                        word_position = target["word_position"]

                        if word_position not in results:
                            raise RuntimeError(
                                "Embedding returned a target position "
                                f"that was not requested: {word_position}"
                            )

                        results[word_position][
                            config["name"]
                        ] = vector

        missing = []

        for position in sorted(target_positions):
            missing_scales = [
                scale
                for scale in SCALES
                if scale not in results[position]
            ]

            if missing_scales:
                missing.append(
                    (
                        position,
                        document.rows[position].token,
                        missing_scales,
                    )
                )

        if missing:
            logger.error(
                "[tier1] incomplete embeddings: %d observations",
                len(missing),
            )

            for position, token, scales in missing[:20]:
                logger.error(
                    "[tier1] position=%d token=%r missing=%s",
                    position,
                    token,
                    scales,
                )

            raise RuntimeError(
                f"{len(missing)} observations did not receive "
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

        word_spans: list[tuple[int, int]] = []
        current_word = None
        current_start = None

        for encoded_position, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if word_id != current_word:
                if current_word is not None:
                    word_spans.append(
                        (
                            current_start,
                            encoded_position,
                        )
                    )

                current_word = word_id
                current_start = encoded_position

        if current_word is not None:
            word_spans.append(
                (
                    current_start,
                    len(word_ids),
                )
            )

        if len(word_spans) != word_count:
            raise RuntimeError(
                "MacBERTh word alignment is incomplete: "
                f"expected {word_count} corpus tokens, "
                f"got {len(word_spans)} encoded spans."
            )

        jobs: list[dict] = []
        covered_targets: set[int] = set()

        start_word = 0

        while start_word < word_count:
            end_word = min(
                word_count,
                start_word + window_size,
            )

            candidate_targets = sorted(
                position
                for position in target_positions
                if start_word <= position < end_word
            )

            if candidate_targets:
                group: list[int] = []

                for target in candidate_targets:
                    if not group:
                        group.append(target)
                        continue

                    group_start = word_spans[group[0]][0]
                    group_end = word_spans[target][1]

                    if group_end - group_start <= 512:
                        group.append(target)
                    else:
                        self._append_job(
                            jobs=jobs,
                            covered_targets=covered_targets,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            word_ids=word_ids,
                            word_spans=word_spans,
                            target_positions=group,
                            context_start_word=start_word,
                            context_end_word=end_word,
                        )

                        group = [target]

                if group:
                    self._append_job(
                        jobs=jobs,
                        covered_targets=covered_targets,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        word_ids=word_ids,
                        word_spans=word_spans,
                        target_positions=group,
                        context_start_word=start_word,
                        context_end_word=end_word,
                    )

            if start_word + stride >= word_count:
                break

            start_word += stride

        missing_targets = target_positions - covered_targets

        if missing_targets:
            raise RuntimeError(
                "Some target observations were not assigned to "
                f"a MacBERTh job: {sorted(missing_targets)[:20]}"
            )

        return jobs

    def _append_job(
        self,
        *,
        jobs: list[dict],
        covered_targets: set[int],
        input_ids: list[int],
        attention_mask: list[int],
        word_ids: list[int | None],
        word_spans: list[tuple[int, int]],
        target_positions: list[int],
        context_start_word: int,
        context_end_word: int,
    ) -> None:
        target_start_word = target_positions[0]
        target_end_word = target_positions[-1] + 1

        context_start = word_spans[
            context_start_word
        ][0]

        context_end = word_spans[
            context_end_word - 1
        ][1]

        target_start = word_spans[
            target_start_word
        ][0]

        target_end = word_spans[
            target_end_word - 1
        ][1]

        target_span = target_end - target_start

        if target_span > 512:
            raise RuntimeError(
                "A target group exceeds MacBERTh's 512-position "
                f"limit: targets={target_start_word}:"
                f"{target_end_word}, encoded_length={target_span}"
            )

        available_length = context_end - context_start

        if available_length > 512:
            desired_start = target_start - (
                512 - target_span
            ) // 2

            encoded_start = max(
                context_start,
                desired_start,
            )

            encoded_end = min(
                context_end,
                encoded_start + 512,
            )

            if encoded_end - encoded_start < 512:
                encoded_start = max(
                    context_start,
                    encoded_end - 512,
                )
        else:
            encoded_start = context_start
            encoded_end = context_end

        if not (
            encoded_start <= target_start
            and target_end <= encoded_end
        ):
            raise RuntimeError(
                "Constructed MacBERTh context does not contain "
                f"all targets: targets={target_start_word}:"
                f"{target_end_word}, "
                f"context={context_start_word}:"
                f"{context_end_word}"
            )

        relative_word_ids = word_ids[
            encoded_start:encoded_end
        ]

        window_ids = input_ids[
            encoded_start:encoded_end
        ].copy()

        window_mask = attention_mask[
            encoded_start:encoded_end
        ]

        target_positions_in_window = []

        for word_position in target_positions:
            try:
                relative = relative_word_ids.index(
                    word_position
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Target disappeared from its MacBERTh "
                    f"window: word_position={word_position}"
                ) from exc

            target_positions_in_window.append(
                {
                    "word_position": word_position,
                    "encoded_position": relative,
                }
            )

            if self.mask_targets:
                mask_token_id = self.tokenizer.mask_token_id

                if mask_token_id is None:
                    raise RuntimeError(
                        "MacBERTh tokenizer has no mask token."
                    )

                window_ids[relative] = mask_token_id

        jobs.append(
            {
                "input_ids": window_ids,
                "attention_mask": window_mask,
                "targets": target_positions_in_window,
            }
        )

        covered_targets.update(target_positions)

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

        # BERT's internal token_type_ids buffer is only 512 tokens long.
        # The job builder therefore guarantees max_length <= 512.
        if max_length > 512:
            raise RuntimeError(
                f"Prepared MacBERTh batch exceeds 512 tokens: {max_length}"
            )

        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            raise RuntimeError(
                "MacBERTh tokenizer has no pad token."
            )

        input_ids = []
        attention_masks = []

        for job in jobs:
            padding = max_length - len(job["input_ids"])

            input_ids.append(
                job["input_ids"] + [pad_token_id] * padding
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
            # Use MacberthModel.encode(), which deliberately calls
            # model.base_model rather than the MaskedLM head.
            output = self.mac.encode(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                return_dict=True,
            )

        hidden = output.last_hidden_state.cpu().numpy()

        return [
            [
                hidden[
                    batch_index,
                    target["encoded_position"],
                ].astype(np.float32, copy=False)
                for target in job["targets"]
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

        new_observations = self._new_observations(
            observations
        )

        if not new_observations:
            return 0

        self._write_postgres(new_observations)
        self._write_lance(new_observations)

        return len(new_observations)

    def repair_lance(
        self,
        observations: list[EmbeddedObservation],
    ) -> int:
        if not observations:
            return 0

        # Repair deliberately does not consult or modify PostgreSQL.
        # The repair unit is a document whose current event records already
        # exist; Lance is reconciled only for the explicitly requested work.
        return self._write_lance(observations)

    def _new_observations(
        self,
        observations: list[EmbeddedObservation],
    ) -> list[EmbeddedObservation]:
        event_ids = [
            embedded.observation.event_id
            for embedded in observations
        ]

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id
                FROM events
                WHERE event_id = ANY(%s)
                """,
                (event_ids,),
            )

            existing_ids = {
                row[0]
                for row in cur.fetchall()
            }

        return [
            embedded
            for embedded in observations
            if embedded.observation.event_id
            not in existing_ids
        ]

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
    ) -> int:
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

        written_event_ids: set[int] = set()

        for table_name, rows in rows_by_table.items():
            table = self._open_table(
                table_name,
                vector_dimensions=len(rows[0]["vector"]),
            )

            existing_ids = self._existing_lance_ids(
                table,
                {
                    row["event_id"]
                    for row in rows
                },
            )

            new_rows = [
                row
                for row in rows
                if row["event_id"] not in existing_ids
            ]

            if new_rows:
                table.add(
                    new_rows,
                    mode="append",
                )

                written_event_ids.update(
                    row["event_id"]
                    for row in new_rows
                )

        return len(written_event_ids)

    def _existing_lance_ids(
        self,
        table,
        event_ids: set[int],
    ) -> set[int]:
        if not event_ids:
            return set()

        arrow = table.to_arrow()

        existing = set(
            arrow.column("event_id").to_pylist()
        )

        return existing.intersection(event_ids)

    def _open_table(
        self,
        table_name: str,
        *,
        vector_dimensions: int,
    ):
        if table_name in self.tables:
            return self.tables[table_name]

        table_names = set(
            self.lance.list_tables().tables
        )

        if table_name in table_names:
            table = self.lance.open_table(table_name)
            self.tables[table_name] = table
            return table

        logger.info(
            "[tier1] creating Lance table: %s",
            table_name,
        )

        table = self.lance.create_table(
            table_name,
            schema={
                "event_id": np.uint64,
                "year": np.int32,
                "embedding_model": str,
                "vector": lancedb.vector(vector_dimensions),
            },
        )

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

        if documents:
            logger.info(
                "[tier1] first seed documents: %s",
                documents[:5],
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

            observations = self._build_observations(
                document,
                target_positions,
                embeddings,
            )

            written = self.writer.write(observations)

            elapsed = time.perf_counter() - started

            logger.info(
                "[tier1] %3d/%-3d %-4s %-15s seeds=%3d observations=%5d written=%5d elapsed=%7.2fs",
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

    def repair(
        self,
        *,
        corpus: str,
        doc_id: str,
    ) -> None:
        started = time.perf_counter()

        logger.info(
            "[tier1] repair: %s/%s",
            corpus,
            doc_id,
        )

        document = self._load_document(
            corpus,
            doc_id,
        )

        if document is None:
            raise RuntimeError(
                f"Document not found: {corpus}/{doc_id}"
            )

        seed_positions = {
            position
            for position, row in enumerate(document.rows)
            if is_seed(row.token)
        }

        if not seed_positions:
            raise RuntimeError(
                f"No seed occurrences found: {corpus}/{doc_id}"
            )

        target_positions = self._select_neighbours(
            seed_positions,
            len(document.rows),
        )

        logger.info(
            "[tier1] repair: seeds=%d observations=%d",
            len(seed_positions),
            len(target_positions),
        )

        embeddings = self.pipeline.embed(
            document,
            target_positions,
        )

        observations = self._build_observations(
            document,
            target_positions,
            embeddings,
        )

        written = self.writer.repair_lance(
            observations,
        )

        elapsed = time.perf_counter() - started

        logger.info(
            "[tier1] repair complete: %-4s %-15s "
            "seeds=%3d observations=%5d lance_written=%5d elapsed=%7.2fs",
            corpus,
            doc_id,
            len(seed_positions),
            len(observations),
            written,
            elapsed,
        )

    def _build_observations(
        self,
        document: DocBuffer,
        target_positions: set[int],
        embeddings: dict[int, dict[str, np.ndarray]],
    ) -> list[EmbeddedObservation]:
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

        return observations

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


def parse_repair_target(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError(
            "repair target must be CORPUS/DOC_ID"
        )

    corpus, doc_id = value.split("/", 1)

    if not corpus or not doc_id:
        raise argparse.ArgumentTypeError(
            "repair target must be CORPUS/DOC_ID"
        )

    return corpus, doc_id


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
        "--repair",
        type=parse_repair_target,
        metavar="CORPUS/DOC_ID",
        help=(
            "Regenerate Lance vectors for one document without "
            "modifying PostgreSQL events."
        ),
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

    args = parser.parse_args()

    if args.repair is not None and (
        args.corpus is not None
        or args.doc_id is not None
    ):
        parser.error(
            "--repair cannot be combined with --corpus or --doc-id"
        )

    return args


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

        if args.repair is not None:
            corpus, doc_id = args.repair

            processor.repair(
                corpus=corpus,
                doc_id=doc_id,
            )
        else:
            processor.process(
                corpus=args.corpus,
                doc_id=args.doc_id,
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
