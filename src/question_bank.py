"""Question Bank — searchable index over pre-prepared JSON question files.

Loads interview_questions.json / genai_question_bank.json into a single searchable corpus and
ranks it with a HYBRID score: semantic cosine (local sentence-transformers embeddings) blended
with lexical TF-IDF cosine.

Why hybrid: pure TF-IDF matches words, not meaning — "F5-TTS audio generation voice cloning"
scored "What is RAG (Retrieval-Augmented Generation)?" as a top hit because both contain
"generation". Embeddings fix recall; TF-IDF is kept as a minority weight because exact tool/API
names ("n8n", "F5-TTS", "LoRA") are precisely where lexical matching beats a small embedding model.

Embeddings are optional: if `src.embeddings` is unavailable the retriever degrades to the previous
pure-TF-IDF behaviour. The corpus embedding matrix is cached on disk (keyed by corpus digest +
model name) so the one-off encode cost is not paid on every process start.

No SQLite, no live scraping — all data comes from data/ JSON files.
"""

import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EMBED_CACHE_DIR, EMBEDDING_MODEL, HYBRID_EMBED_WEIGHT, HYBRID_MIN_SCORE
from src.models import QuestionDetail


class QuestionBankRetriever:
    """Hybrid (semantic + lexical) search over the pre-indexed question bank."""

    def __init__(self, questions: list[dict], cache_key: str = ""):
        self._corpus = questions
        self._cache_key = cache_key
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._embed_matrix: np.ndarray | None = None
        self._build_index()

    # ── Index ────────────────────────────────────────────────────────────────
    def _build_index(self):
        """Build the TF-IDF index and (when available) the cached embedding matrix."""
        if not self._corpus:
            return

        texts = [q.get("content", "") for q in self._corpus]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        self._embed_matrix = self._load_or_build_embeddings(texts)

    def _load_or_build_embeddings(self, texts: list[str]) -> np.ndarray | None:
        """L2-normalized (n×d) corpus embeddings, from disk cache when the corpus is unchanged.
        None when embeddings are unavailable → caller falls back to pure TF-IDF."""
        from src import embeddings

        if not embeddings.available():
            return None

        digest = hashlib.sha256(
            ("\n".join(texts) + "|" + EMBEDDING_MODEL).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = EMBED_CACHE_DIR / f"{self._cache_key or 'bank'}-{digest}.npy"

        try:
            if cache_path.exists():
                cached = np.load(cache_path)
                if cached.shape[0] == len(texts):
                    return cached
        except Exception:  # noqa: BLE001 — a bad cache file must never break retrieval
            pass

        matrix = embeddings.embed(texts)
        if matrix is None:
            return None
        try:
            EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Stale entries for this bank are superseded; drop them so the dir can't grow forever.
            for old in EMBED_CACHE_DIR.glob(f"{self._cache_key or 'bank'}-*.npy"):
                if old != cache_path:
                    old.unlink(missing_ok=True)
            np.save(cache_path, matrix)
        except Exception:  # noqa: BLE001 — caching is best-effort
            pass
        return matrix

    # ── Search ───────────────────────────────────────────────────────────────
    def _scores(self, query: str) -> np.ndarray:
        """Hybrid relevance of every corpus row to `query` (semantic + lexical)."""
        lexical = cosine_similarity(
            self._vectorizer.transform([query]), self._tfidf_matrix
        ).flatten()
        if self._embed_matrix is None:
            return lexical

        from src import embeddings
        q_vec = embeddings.embed([query])
        if q_vec is None:
            return lexical
        # Both matrices are L2-normalized, so the dot product IS cosine similarity.
        semantic = (self._embed_matrix @ q_vec[0]).astype(float)
        w = HYBRID_EMBED_WEIGHT
        return w * semantic + (1.0 - w) * lexical

    def search(
        self,
        query: str,
        difficulty: str | None = None,
        source: str | None = None,
        limit: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[QuestionDetail]:
        """Search by hybrid semantic+lexical cosine similarity with optional filters."""
        if not self._corpus or self._tfidf_matrix is None:
            return []

        scores = self._scores(query)
        # A pure-lexical run keeps the historical near-zero floor; a hybrid run uses a real floor
        # because embedding cosine is meaningfully non-zero even for unrelated text.
        floor = HYBRID_MIN_SCORE if self._embed_matrix is not None else 0.01

        candidates = []
        for i, row in enumerate(self._corpus):
            q_id = row.get("id", "")
            if exclude_ids and q_id in exclude_ids:
                continue
            if difficulty and row.get("difficulty") and row["difficulty"] != difficulty:
                continue
            if source and row.get("source") and row["source"] != source:
                continue
            if scores[i] < floor:
                continue
            candidates.append((scores[i], row))

        candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, row in candidates[:limit]:
            results.append(QuestionDetail(
                question_id=row.get("id", ""),
                category=row.get("category", row.get("question_type", "GENERAL")),
                content=row.get("content", ""),
                topic=row.get("topic", ""),
                sub_topic=row.get("sub_topic"),
                difficulty=row.get("difficulty", "Medium"),
                asked_in_company=row.get("company"),
                role=row.get("role"),
                source_url=row.get("source_url"),
                source="interview_db",
                retrieval_score=round(float(score), 4),
            ))

        return results

    def index_kind(self) -> str:
        """Which ranking is actually active — useful in logs/eval to prove the semantic path ran."""
        return "hybrid semantic+TF-IDF" if self._embed_matrix is not None else "TF-IDF only"

    def get_stats(self) -> dict:
        """Get question bank statistics."""
        if not self._corpus:
            return {"total": 0}

        sources = {}
        difficulties = {}
        for q in self._corpus:
            s = "interview_db"
            sources[s] = sources.get(s, 0) + 1
            d = q.get("difficulty", "Medium")
            difficulties[d] = difficulties.get(d, 0) + 1

        return {
            "total": len(self._corpus),
            "by_source": sources,
            "by_difficulty": difficulties,
            "index": self.index_kind(),
        }


# ── Singleton ───────────────────────────────────────────────────────────────

_retriever: QuestionBankRetriever | None = None
_genai_retriever: QuestionBankRetriever | None = None


def get_retriever() -> QuestionBankRetriever:
    """Get or create the default question bank retriever (Python/SWE interview data)."""
    global _retriever
    if _retriever is None:
        from src.data_loader import get_data_store
        data_store = get_data_store()
        all_questions = data_store.get_all_questions()
        _retriever = QuestionBankRetriever(all_questions, cache_key="interview")
        print(f"Question bank ready: {len(all_questions)} questions indexed "
              f"({_retriever.index_kind()} search)")
    return _retriever


def get_genai_retriever() -> QuestionBankRetriever:
    """Curated GenAI question bank (built by scripts/build_genai_bank.py). Empty until built."""
    global _genai_retriever
    if _genai_retriever is None:
        import json
        from src.config import GENAI_BANK_JSON
        questions: list[dict] = []
        if GENAI_BANK_JSON.exists():
            try:
                questions = json.loads(GENAI_BANK_JSON.read_text(encoding="utf-8"))
            except Exception:
                questions = []
        _genai_retriever = QuestionBankRetriever(questions, cache_key="genai")
        print(f"GenAI bank ready: {len(questions)} questions indexed "
              f"({_genai_retriever.index_kind()} search)")
    return _genai_retriever


def get_retriever_for(category: str | None) -> QuestionBankRetriever:
    """Route bank retrieval by domain: GEN_AI sessions use the curated GenAI bank (if built),
    everything else uses the default Python/SWE bank."""
    if (category or "").upper() == "GEN_AI":
        r = get_genai_retriever()
        if r._corpus:                    # only if the GenAI bank has been built
            return r
    return get_retriever()
