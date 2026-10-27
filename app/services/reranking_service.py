"""Optional cross-encoder reranking (CPU). Loaded lazily on first use."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker %s (CPU)", self.model_name)
                    self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages], show_progress_bar=False)]
