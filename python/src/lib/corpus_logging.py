# corpus_logging.py

import os
import sys
import json
import logging
from logging.handlers import TimedRotatingFileHandler

from typing import Any, Callable
from lib.corpus_config import LOG_DIR

EmitFn = Callable[[str, str], None]

logger = logging.getLogger("corpus")

level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
logging_level = getattr(logging, level_name, logging.INFO)
logger.setLevel(logging_level)


if not logger.handlers:
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    formatter = logging.Formatter(
        "%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _h = logging.StreamHandler(sys.stderr)
    _h.setLevel(logging_level)
    _h.setFormatter(formatter)
    logger.addHandler(_h)

    _h = TimedRotatingFileHandler(
        LOG_DIR / "corpus.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(formatter)
    logger.addHandler(_h)


class CorpusLogger(logging.LoggerAdapter):
    def __init__(
        self,
        logger: logging.Logger,
        emit: EmitFn | None = None,
        tag: str = "",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(logger, {})
        self._emit = emit
        self._tag = tag
        self._context = self._normalise_context(context)

    @staticmethod
    def _normalise_context(context) -> dict:
        """
        Coerce whatever was passed as context into a plain dict safe for
        use with dict.update().

        Callers sometimes pass a string, a list of tuples, or other
        non-dict values. Rather than requiring every call site to be
        correct, we absorb the mismatch here and store a JSON-serialisable
        dict in all cases:

            None                 -> {}
            dict                 -> dict (used as-is)
            str                  -> {"context": "<value>"}
            list/tuple of pairs  -> dict(value)   e.g. [("a", 1)] -> {"a": 1}
            anything else        -> {"context": str(value)}
        """
        if not context:
            return {}
        if isinstance(context, dict):
            return context
        if isinstance(context, str):
            return {"context": context}
        if isinstance(context, (list, tuple)):
            try:
                return dict(context)
            except (TypeError, ValueError):
                return {"context": json.dumps(context, default=str)}
        return {"context": str(context)}

    def process(self, msg, kwargs):
        if self._tag:
            msg = f"{self._tag} {msg}"
        kwargs.setdefault("extra", {}).update(self._context)
        return msg, kwargs

    def log(self, level, msg, *args, **kwargs):
        if self._emit:
            try:
                rendered = msg % args if args else str(msg)
            except Exception:
                rendered = str(msg)

            payload = dict(self._context)
            payload.update(kwargs.get("extra", {}))

            self._emit(
                rendered,
                json.dumps(payload, default=str),
            )

        super().log(level, msg, *args, **kwargs)


def setEmit(
    emit: EmitFn,
    tag: str = "",
    context: dict[str, Any] | None = None,
) -> CorpusLogger:
    return CorpusLogger(
        logger,
        emit=emit,
        tag=tag,
        context=context,
    )
