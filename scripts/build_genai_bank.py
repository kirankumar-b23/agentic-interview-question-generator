#!/usr/bin/env python3
"""Curate a GenAI interview-question bank from the web (one-time / periodic).

The default interview bank (data/interview_questions.json) is a Python/SWE set with almost no
GenAI content. This harvests REAL GenAI interview questions across a curated list of role/concept
queries via the existing Tavily connector, de-duplicates them (embeddings, else normalized text),
keeps a credible company where extracted (company-only bank), tags each question's ROLE, records
CROSS-SOURCE corroboration, and merges a small hand-verified SEED set. Writes
data/genai_question_bank.json in the shape src/question_bank.py expects.

Run:  python scripts/build_genai_bank.py        (needs TAVILY_API_KEY)
"""
from __future__ import annotations
import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENAI_BANK_JSON, TAVILY_API_KEY   # noqa: E402
from src.sources.base import domain                       # noqa: E402
from src.sources.tavily_search import TavilyConnector     # noqa: E402

# Curated GenAI queries — concepts, role-specific, and topics the report flagged we were thin on.
QUERIES = [
    # foundations / capabilities
    "large language models", "LLM fundamentals", "how LLMs work",
    "open source vs closed source LLM", "pre-trained vs fine-tuned models",
    "multimodal LLM", "Hugging Face models", "tokens in language models",
    "LLM context window", "temperature top-p LLM", "generative AI capabilities",
    "AI model selection reasoning vs fast models",
    # prompting
    "prompt engineering techniques", "chain of thought prompting",
    "few-shot vs zero-shot prompting", "system prompt design", "prompt engineering best practices",
    # RAG / agents / tools
    "retrieval augmented generation RAG", "vector embeddings", "vector database",
    "AI agents", "LangChain agents", "agent memory", "multi agent systems",
    "model context protocol MCP", "tool use function calling LLM",
    # fine-tuning / eval / media
    "fine-tuning LLM basics", "when to fine-tune vs prompt engineering",
    "LLM evaluation metrics", "hallucination in LLMs",
    "AI image generation", "stable diffusion", "text to speech AI",
    # report-derived: topics we were thin on
    "responsible AI interview", "AI ethics and bias interview",
    "reduce hallucinations LLM interview", "enterprise RAG interview",
    "LLM guardrails safety interview", "LLM system design interview",
    # report-derived: role-specific
    "generative AI engineer interview questions", "prompt engineer interview questions",
    "machine learning engineer LLM interview questions", "AI data scientist generative interview",
    "generative AI product manager interview questions",
]
CHUNK = 12          # TavilyConnector caps outcomes per fetch; harvest in chunks
DEDUP_SIM = 0.90    # cosine ≥ ⇒ near-duplicate

# ── Hand-verified seed questions (deep-research report §1) — real, company + role + difficulty ──
SEED_QUESTIONS = [
    ("What is Generative AI, and how is it different from traditional AI or Machine Learning?",
     "Capgemini", "ML/AI Engineer", "Easy"),
    ("What is a transformer model, and what is the attention mechanism?",
     "Capgemini", "ML/AI Engineer", "Medium"),
    ("Why is prompt engineering important, and what makes a good prompt? Give examples of poor vs good prompts.",
     "Capgemini", "Prompt Engineer", "Easy"),
    ("What is Responsible AI? What are the ethical risks of Generative AI, and how does AI bias occur?",
     "Capgemini", "ML/AI Engineer", "Medium"),
    ("What are hallucinations in LLMs, and how can they be reduced?",
     "Capgemini", "ML/AI Engineer", "Medium"),
    ("What is Retrieval-Augmented Generation (RAG), why is it needed in enterprise GenAI, and how does it improve accuracy?",
     "Capgemini", "ML/AI Engineer", "Medium"),
    ("How would you validate the output of an AI model, and how would you handle incorrect or biased responses?",
     "Capgemini", "ML/AI Engineer", "Medium"),
    ("What is the meaning of RAG (Retrieval-Augmented Generation)?",
     "Cognizant", "ML/AI Engineer", "Easy"),
    ("Design a prompt for booking a meeting that includes why to book, where to book, the name, and other necessary information.",
     "Nextiva", "Prompt Engineer", "Medium"),
    ("What approaches have you tested for implementing a Retrieval-Augmented Generation (RAG) system?",
     "Schneider Electric", "Data Scientist", "Hard"),
    ("Explain RAG and LoRA, and how you would apply them in a Generative AI solution.",
     "Amazon", "Data Scientist", "Hard"),
]


# Canonical display casing for common companies (merges "Openai"/"OpenAI", "Servicenow"→"ServiceNow").
_CANON = {
    "openai": "OpenAI", "huggingface": "Hugging Face", "hugging": "Hugging Face",
    "servicenow": "ServiceNow", "bytedance": "ByteDance", "tiktok": "TikTok",
    "jpmorganchase": "JPMorgan Chase", "hcltech": "HCLTech", "nvidia": "NVIDIA",
    "sap": "SAP", "rbc": "RBC", "ibm": "IBM", "aig": "AIG", "pwc": "PwC",
    "langchain": "LangChain", "characterai": "Character AI",
}


def _canon(company: str | None) -> str | None:
    if not company:
        return company
    return _CANON.get(company.strip().lower(), company.strip())


def _infer_role(text: str) -> str:
    """Keyword heuristic → the role a question most fits (report's role dimension)."""
    t = text.lower()
    if any(k in t for k in ("prompt", "chain of thought", "few-shot", "zero-shot", "system prompt")):
        return "Prompt Engineer"
    if any(k in t for k in ("roadmap", "prioriti", "product manager", "go-to-market",
                            "stakeholder", "product strategy", "what should we build")):
        return "GenAI Product Manager"
    if any(k in t for k in ("rag", "retrieval", "agent", "embedding", "fine-tun", "deploy",
                            "vector", "langchain", "function calling", "pipeline", "system design",
                            "architecture", "latency", "scale")):
        return "ML/AI Engineer"
    if any(k in t for k in ("statistic", "probability", "evaluation", "metric", "dataset",
                            "data analysis", "experiment", "a/b", "distribution")):
        return "Data Scientist"
    return "General"


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _dedup_with_corroboration(records: list) -> list:
    """Collapse duplicates, keeping a source_count = # distinct source domains a question
    appeared across (the report's cross-source authenticity signal). Prefers the company-
    bearing variant. Returns list of dicts {record, source_count}."""
    groups: dict = defaultdict(lambda: {"rep": None, "domains": set()})
    for r in records:
        key = _norm(r.question_text)
        if not key:
            continue
        g = groups[key]
        g["domains"].add(domain(r.source_url) or r.source_type)
        # Prefer a representative that carries a real company.
        if g["rep"] is None or (not g["rep"].company and r.company):
            g["rep"] = r
    items = [{"record": g["rep"], "source_count": len(g["domains"])} for g in groups.values()]

    # Semantic dedup across representatives; merge source_counts + keep company-bearing.
    try:
        from src import embeddings
        texts = [it["record"].question_text for it in items]
        sim = embeddings.cosine_matrix(texts)
        if sim is not None:
            drop = set()
            for i in range(len(items)):
                if i in drop:
                    continue
                for j in range(i + 1, len(items)):
                    if j not in drop and sim[i][j] >= DEDUP_SIM:
                        items[i]["source_count"] += items[j]["source_count"]
                        if not items[i]["record"].company and items[j]["record"].company:
                            items[i]["record"] = items[j]["record"]
                        drop.add(j)
            items = [it for k, it in enumerate(items) if k not in drop]
    except Exception as e:  # noqa: BLE001
        print(f"[dedup] embeddings unavailable ({e}); used text dedup only")
    return items


def _apply_form_gate(bank: list[dict]) -> tuple[list[dict], int]:
    """Strip scrape artifacts and drop rows that aren't well-formed standalone questions.

    Re-validates the company on harvested (`web`) rows too, so a site-brand fragment ("Tech" from
    techinterviewhandbook.org) is blanked to an honest source label instead of being presented as an
    employer. Curated seed/xlsx names are trusted as-is. Returns (kept_rows, dropped_count).
    """
    from src.quality import is_quality_question, strip_artifacts
    from src.sources.tavily_search import _valid_company

    kept, dropped = [], 0
    for q in bank:
        content = strip_artifacts(q.get("content", ""))
        if not is_quality_question(content):
            dropped += 1
            continue
        q["content"] = content
        if q.get("source") == "web" and q.get("company"):
            q["company"] = _valid_company(q["company"])
        kept.append(q)
    return kept, dropped


def main() -> int:
    if not TAVILY_API_KEY:
        print("ERROR: TAVILY_API_KEY not set — cannot harvest.")
        return 1

    all_records = []
    connector = TavilyConnector()
    for i in range(0, len(QUERIES), CHUNK):
        chunk = QUERIES[i:i + CHUNK]
        print(f"harvesting queries {i + 1}..{i + len(chunk)} of {len(QUERIES)} …", flush=True)
        records, calls, err = connector.fetch(chunk)
        if err:
            print(f"  tavily note: {err}")
        all_records.extend(records)
        print(f"  +{len(records)} records (total {len(all_records)})")

    items = _dedup_with_corroboration(all_records)

    # Harvested questions — company kept when a credible one was extracted (junk already filtered by
    # _valid_company), otherwise left empty. No-company questions are now KEPT (team decision): they are
    # still valuable for prep and are attributed to their SOURCE SITE at output time (see models.attribution_label).
    seed_keys = {_norm(c) for c, *_ in SEED_QUESTIONS}
    harvested = [{
        "id": str(uuid.uuid4()),
        "content": it["record"].question_text,
        "topic": "Gen AI",
        "difficulty": "Medium",                     # re-tagged at runtime by validate_relevance
        "company": _canon(it["record"].company),    # may be None → source-site attribution downstream
        "role": _infer_role(it["record"].question_text),
        "source": "web",
        "source_url": it["record"].source_url,
        "source_count": it["source_count"],
    } for it in items if _norm(it["record"].question_text) not in seed_keys]

    # Hand-verified seed (report §1) — always included, marked verified.
    seed = [{
        "id": str(uuid.uuid4()),
        "content": content,
        "topic": "Gen AI",
        "difficulty": difficulty,
        "company": company,
        "role": role,
        "source": "seed",
        "source_url": "",
        "source_count": 3,                          # hand-verified → treat as corroborated
    } for content, company, role, difficulty in SEED_QUESTIONS]

    # Corroboration-first ordering: seed, then multi-source, then the rest.
    bank = seed + sorted(harvested, key=lambda q: q["source_count"], reverse=True)

    # FORM gate at build time. The Tavily extractor's own filters let page furniture through —
    # blog titles ("Is GPT Image 2 the Best Image Generation Model?"), SEO tails ("… | Dataford
    # Interview Questions"), pronoun fragments ("how does it affect generation?") — which then had
    # to be swept up afterwards by scripts/clean_bank.py. Gating here means a rebuild cannot
    # reintroduce them. Same functions the runtime and the cleaner use, so the rules stay in sync.
    bank, form_dropped = _apply_form_gate(bank)

    GENAI_BANK_JSON.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    roles = Counter(q["role"] for q in bank)
    corrob = sum(1 for q in bank if q["source_count"] >= 2)
    print(f"\nWrote {len(bank)} GenAI questions → {GENAI_BANK_JSON}")
    print(f"  seed: {len(seed)} | harvested: {len(harvested)} | corroborated (≥2 sources): {corrob}")
    print(f"  dropped by form gate: {form_dropped}")
    print(f"  with real company: {sum(1 for q in bank if q.get('company'))} / {len(bank)}")
    print(f"  roles: {dict(roles)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
