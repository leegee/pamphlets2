from __future__ import annotations

"""
Experiment: diachronic semantic search using a MacBERTh phrase query.

Run with:

    python -m retrieval.tests.experiments.test_phrase_search2
"""

from lib.corpus_config import EVENTSTORE_T1_PATH
from lib.corpus_logging import logger
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.macberth_phrase_encoder2 import MacBertMeanPhraseEncoder
from retrieval.models import SearchSpace
from retrieval.observation_retriever import IndexedObservationRetriever
from retrieval.parquet_context import ParquetContext

YEAR = None
SCALE = None
K = 50
PHRASE = "white as wool"
CARRIER = "They had {}."

space = SearchSpace(
    years=YEAR,
    scale=SCALE,
)


def main() -> None:
    encoder = MacBertMeanPhraseEncoder()

    query = encoder.encode(
        PHRASE,
        carrier=CARRIER,
    )

    index_store = LanceObservationIndexStore()

    context = ParquetContext(
        EVENTSTORE_T1_PATH,
        context_before=10,
        context_after=10,
    )

    retriever = IndexedObservationRetriever(
        index_store=index_store,
        context=context,
    )

    results = retriever.diachronic_search(
        query,
        space=space,
        k=K,
        direction="forward",
    )

    logger.info("")
    logger.info("DIACHRONIC PHRASE SEARCH")
    logger.info("=" * 70)
    logger.info(f"phrase:     {PHRASE}")
    logger.info(f"encoder:    {type(encoder).__name__}")
    logger.info(f"carrier:    {CARRIER}")
    logger.info(f"year:       {YEAR if YEAR is not None else 'ALL'}")
    logger.info(f"scale:      {SCALE if SCALE is not None else 'ALL'}")
    logger.info(f"k:          {K}")
    logger.info(f"direction:  forward")
    logger.info("." * 70)

    logger.info("")
    logger.info("RESULTS")
    logger.info("=" * 70)


    all_results = []


    for (bucket_start, bucket_end), bucket_results in results:
        all_results.extend(bucket_results)

        logger.info("")
        logger.info( f"BUCKET {bucket_start}-{bucket_end}" )
        logger.info("-" * 70)

        for rank, result in enumerate(bucket_results[:5], start=1):
            observation = result.observation

            logger.info(
                f"{rank:>2}. "
                f"{result.distance:.6f} "
                f"{result.event_id} "
                f"{observation['doc_id']} "
                f"{observation['token']!r}"
            )

            logger.info(
                f"    {result.text}"
            )

    all_results.sort(
        key=lambda result: result.distance,
    )

    logger.info("")
    logger.info("")
    logger.info("GLOBAL TOP 50")
    logger.info("=" * 70)

    for rank, result in enumerate(all_results[:50], start=1):
        observation = result.observation

        logger.info(
            f"{rank:>2}. "
            f"{result.distance:.6f} "
            f"{result.event_id} "
            f"{observation['doc_id']} "
            f"{observation['token']!r}"
        )

        logger.info(
            f"    {result.text}"
        )

    logger.info("." * 70)


if __name__ == "__main__":
    main()
