"""Semantic embeddings for retrieval, redundancy (MMR), coverage, and session attribution.

Uses a local sentence-transformers model (free, no API key) — default all-MiniLM-L6-v2,
which is small (~80MB) and cached under ~/.cache/huggingface. Runs fully offline
(HF_HUB_OFFLINE) since the model is pre-downloaded.

Everything degrades gracefully: if sentence-transformers/torch aren't installed or the
model can't load, `embed()` returns None and callers fall back to their TF-IDF path — so
the pipeline never hard-fails on the embedding layer.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import numpy as np

from src.config import EMBEDDING_MODEL, EMBEDDINGS_ENABLED

_UNAVAILABLE = False   # set True after a failed load so we don't retry every call


@lru_cache(maxsize=1)
def _model():
    """Load the sentence-transformers model once (offline). None if unavailable."""
    global _UNAVAILABLE
    if _UNAVAILABLE or not EMBEDDINGS_ENABLED:
        return None
    # Force offline: the model is cached; the HF hub host may be blocked.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:  # noqa: BLE001 — any failure → fall back to TF-IDF
        print(f"[embeddings] unavailable ({type(e).__name__}: {e}); falling back to TF-IDF")
        _UNAVAILABLE = True
        return None


def available() -> bool:
    return _model() is not None


def embed(texts: list[str]) -> Optional[np.ndarray]:
    """L2-normalized embedding matrix (n×d) for `texts`, or None if unavailable.
    Normalized so a plain dot product equals cosine similarity."""
    if not texts:
        return None
    m = _model()
    if m is None:
        return None
    try:
        return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    except Exception as e:  # noqa: BLE001
        print(f"[embeddings] encode failed ({type(e).__name__}: {e}); falling back")
        return None


def cosine_matrix(a_texts: list[str], b_texts: list[str] | None = None) -> Optional[np.ndarray]:
    """Cosine-similarity matrix between a_texts and b_texts (or a_texts×a_texts).
    Returns None if embeddings are unavailable — caller should fall back to TF-IDF."""
    a = embed(a_texts)
    if a is None:
        return None
    if b_texts is None:
        return a @ a.T
    b = embed(b_texts)
    if b is None:
        return None
    return a @ b.T
