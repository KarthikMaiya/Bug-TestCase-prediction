from __future__ import annotations

import logging
import torch
from typing import Iterable, List, Tuple

try:
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception:  # pragma: no cover - runtime import
    CrossEncoder = None  # type: ignore


class Reranker:
    """Cross-encoder wrapper for reranking testcase candidates.

    Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` by default and runs on GPU when available.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        if CrossEncoder is None:
            raise RuntimeError("sentence_transformers.cross_encoder.CrossEncoder is unavailable")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CrossEncoder %s on device=%s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)

    def score_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[float]:
        """Score a sequence of (query_text, testcase_title) pairs.

        Returns a list of float scores (higher is better). These are raw model scores; caller
        can apply `torch.sigmoid` or other normalization as desired.
        """
        # CrossEncoder.predict accepts list of pairs
        pair_list = list(pairs)
        if not pair_list:
            return []
        # model.predict returns numpy array
        scores = self.model.predict(pair_list, show_progress_bar=False)
        return [float(s) for s in scores]
