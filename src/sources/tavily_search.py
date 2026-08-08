"""Tavily search connector — breadth + attribution layer for real interview questions.

For each topic outcome it runs a Tavily search, keeps only results whose domain is on
INTERVIEW_SOURCE_ALLOWLIST, and extracts question-like lines from each result's page text.
This single connector reaches every allowlisted source (Glassdoor, AmbitionBox, GeeksforGeeks,
Levels, Blind, Reddit, etc.) legitimately via search — no direct scraping of anti-bot sites.

`search_question()` powers an optional research loop: given a confirmed question, find more
attribution for it. Company attribution is best-effort here; validate_relevance confirms fitness.

Adapted from nxtmock New_AddOn_files/tavily_search.py.
"""
from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse

from src.sources.base import Record, domain, looks_like_question
from src import config

_MAX_OUTCOMES = config.TAVILY_MAX_OUTCOMES
_PER_RESULT = 8
_MAX_RECORDS = config.TAVILY_MAX_RECORDS
_ATTRIBUTION_DOMAINS = ["glassdoor.com", "ambitionbox.com", "tryexponent.com", "datalemur.com",
                        "levels.fyi", "interviewquery.com", "prepfully.com", "igotanoffer.com",
                        "teamblind.com", "leetcode.com", "1point3acres.com",
                        "comparably.com", "bigtechinterviews.com",
                        # High-signal, company-tagged GenAI/ML interview sources
                        "hellointerview.com", "indeed.com", "dataford.io"]
_BROAD_DOMAINS = [
    "glassdoor.com", "ambitionbox.com", "tryexponent.com", "datalemur.com",
    "levels.fyi", "interviewquery.com", "prepfully.com", "igotanoffer.com",
    "teamblind.com", "leetcode.com", "1point3acres.com",
    "stackoverflow.com", "github.com",
    "geeksforgeeks.org", "hackerrank.com",
    "interviewbit.com", "workat.tech", "hackerearth.com", "freecodecamp.org", "careercup.com",
    # GenAI/ML Q&A + forums, SWE interview-experience, extra attribution
    "ai.stackexchange.com", "huggingface.co", "kaggle.com", "machinelearningmastery.com",
    "techinterviewhandbook.org", "neetcode.io", "taro.co", "bigtechinterviews.com",
    "mlstack.cafe", "comparably.com", "fishbowlapp.com",
    # High-signal company-tagged GenAI/ML sources + practitioner roundups
    "hellointerview.com", "indeed.com", "dataford.io", "towardsai.net", "prachub.com",
]


def _client():
    from tavily import TavilyClient
    return TavilyClient(api_key=config.TAVILY_API_KEY)


def _on_allowlist(dom: str, allow: set) -> bool:
    return bool(allow) and any(dom == d or dom.endswith("." + d) for d in allow)


def _company_from_url(url: str) -> Optional[str]:
    """Best-effort company name from URLs that embed it (Glassdoor, AmbitionBox, Levels, etc.)."""
    from urllib.parse import parse_qs
    net = domain(url)
    parsed = urlparse(url)
    path = parsed.path
    if "glassdoor." in net:
        m = re.search(r"/Interview/([A-Za-z0-9\-]+?)-Interview-Questions", path)
        if m:
            return m.group(1).replace("-", " ").title()
    if net.endswith("ambitionbox.com"):
        m = re.search(r"/(?:interviews|overview)/([a-z0-9\-]+?)-(?:interview-questions|interviews)", path)
        if m:
            return m.group(1).replace("-", " ").title()
    if net.endswith("levels.fyi"):
        m = re.search(r"/companies/([a-z0-9\-]+)/", path)
        if m:
            return m.group(1).replace("-", " ").title()
    if net.endswith("tryexponent.com"):
        # Try role-specific page first: /guides/<company>-<role>
        m = re.search(r"/guides/([a-z0-9\-]+?)-(?:data|machine|software|ml|ai|product|senior|engineer|backend|frontend|fullstack|devops|cloud)", path)
        if m:
            return m.group(1).replace("-", " ").title()
        # Broader fallback: /guides/<company>/ or /guides/<company>-interview
        m = re.search(r"/guides/([a-z0-9][a-z0-9\-]*?)(?:/|-interview)", path)
        if m:
            slug = m.group(1)
            if slug not in {"data", "machine", "software", "ml", "ai", "interview"}:
                return slug.replace("-", " ").title()
    if net.endswith("interviewquery.com"):
        params = parse_qs(parsed.query)
        company = params.get("company", [None])[0]
        if company:
            return company.replace("-", " ").title()
    if net.endswith("datalemur.com"):
        m = re.search(r"/(?:sql-interview-questions|interview-questions)/([a-z0-9\-]+?)-[a-z]", path)
        if m:
            return m.group(1).replace("-", " ").title()
    if net.endswith("dataford.io"):
        # /interview-guides/<company>[/<role>] — grab the company slug (role optional).
        m = re.search(r"/interview-guides/([a-z0-9\-]+?)(?:/|$)", path)
        if m:
            return m.group(1).replace("-", " ").title()
    if "indeed.com" in net:
        # /cmp/<Company>/interviews  or  /cmp/<Company>/faq/interviews
        m = re.search(r"/cmp/([A-Za-z0-9\-]+?)/(?:interviews|faq)", path)
        if m:
            return m.group(1).replace("-", " ").title()
    return None


def _company_from_hellointerview(text: str) -> Optional[str]:
    """hellointerview.com puts the company in body copy, not the URL slug —
    e.g. '... system design interview question from Meta and Higgsfield'."""
    m = re.search(r"\bfrom ([A-Z][A-Za-z0-9&.\- ]+?)(?:\.|,| and | \(|$)", text[:400])
    return m.group(1).strip() if m else None


_NOT_COMPANY = {
    "machine", "learning", "data", "science", "deep", "generative", "ai", "ml", "llm", "nlp", "genai",
    "computer", "vision", "coding", "technical", "system", "design", "top", "common", "basic", "advanced",
    "senior", "junior", "the", "python", "sql", "java", "javascript", "react", "statistics", "probability",
    "behavioral", "hr", "software", "engineer", "engineering", "developer", "scientist", "analyst", "mock",
    "sample", "frequently", "asked", "popular", "best", "latest", "real", "fresher", "experienced", "job",
    "rag", "agent", "agentic", "interview", "questions", "answers", "guide", "preparation", "prep",
    "mle", "sde", "swe", "pm", "ds", "da", "llms", "gen", "intelligence", "artificial", "neural", "network",
    "prompt", "prompts", "prompting", "engineering",
    # generic roles / seniority levels — NOT companies (Glassdoor listing pages embed these)
    "manager", "lead", "associate", "consultant", "specialist", "architect", "intern",
    "trainee", "executive", "officer", "head", "director", "principal", "staff", "vp",
    "l1", "l2", "l3", "fullstack", "backend", "frontend", "devops", "cloud", "qa",
    "tester", "support", "graduate", "entry", "level", "role", "position",
    # page UI / navigation noise that precedes the word "interview" on web pages
    "share", "save", "saved", "apply", "follow", "following", "jobs", "job", "salary",
    "salaries", "reviews", "review", "photos", "benefits", "home", "login", "log",
    "sign", "menu", "search", "posted", "read", "more", "view", "see", "all", "your",
    "company", "companies", "overview", "about", "contact", "help", "blog", "news",
}


# Bare legal suffixes / filler that are NOT a company name on their own.
_JUNK_COMPANY = {
    "inc", "inc.", "llc", "ltd", "ltd.", "pvt", "pvt.", "limited", "corp", "corp.",
    "co", "co.", "gmbh", "plc", "technologies", "technology", "solutions", "systems",
    "services", "labs", "group", "global", "pvt ltd", "private limited",
}

# LLM/model names — never a company on their own.
_MODEL_NAMES = {
    "gpt", "gpt-3", "gpt-4", "gpt-4o", "gpt4", "chatgpt", "llama", "llama2", "llama3",
    "gemini", "claude", "bert", "roberta", "mistral", "falcon", "palm", "t5", "dalle",
    "dall-e", "midjourney", "bard", "copilot",
}

_MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct",
           "nov", "dec", "january", "february", "march", "april", "june", "july",
           "august", "september", "october", "november", "december"}

# Job-title / interview-prep / legal tokens — STRIPPED from the ends of a name so
# "Meta Production" → "Meta", "Openai Forward Deployed" → "Openai", "... Interview Questions" → "...".
# (Kept out of here: crack/face/open/… — those only appear as standalone junk, handled by
#  _WHOLE_REJECT, so multi-word names like "Hugging Face" survive.)
_STRIP_ENDS = _JUNK_COMPANY | _MODEL_NAMES | _MONTHS | {
    "interview", "interviews", "question", "questions", "answer", "answers", "guide",
    "overview", "introduction", "conclusion", "summary", "notes", "note", "tips", "faq",
    "prep", "preparation", "role", "roles", "engineer", "developer", "scientist", "analyst",
    "manager", "onsite", "forward", "deployed", "production", "remote", "round", "rounds",
    "phone", "screen", "screening", "coding", "technical", "behavioral", "consultancy",
    "personnel", "staffing", "recruitment", "hiring", "careers", "jobs", "job",
    "cracking", "improving", "example", "examples", "sample",
    # role-title tokens that leak onto real names ("Bytedance Llm Specialist" → "Bytedance",
    # "Vector Marketing Sales Representative" → "Vector Marketing", "Servicenow Research" → …)
    "specialist", "representative", "executive", "extern", "account", "sales", "marketing",
    "research", "software", "ml", "llm", "enterprise", "corporate", "division", "generative",
    "agent",
}

# Source/site brands — the site, not the hiring company (Dataford/HelloInterview/Indeed/…).
_SOURCE_BRANDS = {
    "dataford", "hello", "hellointerview", "indeed", "glassdoor", "ambitionbox", "leetcode",
    "geeksforgeeks", "interviewbit", "prepinsta", "indiabix", "naukri", "quora", "reddit",
    "medium", "kaggle", "exponent", "tryexponent", "datalemur", "levels", "prepfully",
    "igotanoffer", "comparably", "fishbowl", "fishbowlapp", "careercup", "workat",
    "hackerearth", "hackerrank", "freecodecamp", "bigtechinterviews", "mlstack", "taro",
    "neetcode", "techinterviewhandbook", "machinelearningmastery", "towardsai", "prachub",
    "stratascratch", "teamblind", "blind", "simplilearn", "edureka", "intellipaat", "scaler",
    "educative", "testbook", "prepfully",
}

# Tool/framework/protocol/field names + concept fragments that are NOT hiring companies.
_TOOL_FIELD_JUNK = {
    "langchain", "llamaindex", "mcp", "vector", "concept", "multi-agent", "multi agent",
    "natural language processing", "nlp", "ai-enabled", "scenario-based", "meeting protocol",
    "ai first", "decode protocol", "becoming", "os", "my", "every", "pytorch", "tensorflow",
    "kubernetes", "docker", "embedding", "embeddings", "transformer", "transformers", "rag",
    "llms", "diffusion", "crewai", "autogen",
    # tech terms wrongly extracted as companies (seen in bank audit)
    "fine-tuning", "fine tuning", "finetuning", "rest api", "rest apis", "api", "apis",
    "prompt engineering", "machine learning", "deep learning", "generative ai", "gen ai",
    "genai", "gpt", "bert", "lora", "peft", "http", "sql", "python", "javascript",
}

# Reject if the WHOLE cleaned name is one of these: models, standalone junk fragments,
# source brands, tool/field terms, or a single generic/role word.
_WHOLE_REJECT = _MODEL_NAMES | _SOURCE_BRANDS | _NOT_COMPANY | _TOOL_FIELD_JUNK | {
    "crack", "face", "onsite", "open", "improve", "most", "practical", "traditional",
    "modern", "various", "general", "large", "small", "deep",
    # states / generic non-company words seen in extractions
    "offline", "online", "remote", "virtual", "hybrid", "unknown", "none", "other",
    "anonymous", "confidential", "startup", "employer", "freelance", "self", "n/a",
    # Bare-word extraction artifacts observed in the built bank: "Tech" is the truncated
    # techinterviewhandbook.org brand, the rest are page-topic words mistaken for an employer.
    # Whole-string matches only, so real multi-word names ("RedFerns Tech") are unaffected.
    "tech", "product", "classification", "ensemble",
}


def _valid_company(name: Optional[str]) -> Optional[str]:
    """Return a cleaned company name only if it looks like a REAL company, else None.

    Strips trailing/leading job/title/legal/model/date tokens, then rejects the result if it
    is a model name, a source-site brand ("Dataford"/"Hello"), a standalone junk fragment
    ("GPT"/"Crack"/"Face"), a single character ("A"), or an all-generic name. Multi-word real
    names ("Hugging Face") survive. Callers fall back to an honest source label on None."""
    if not name or not any(ch.isalpha() for ch in name):
        return None
    tokens = [t for t in re.split(r"\s+", name.strip()) if t]

    def _norm(t):
        return t.lower().strip(".,&-()")

    def _rem_ok(toks):
        # Don't strip into an empty or whole-rejected remainder (keeps "Vector Marketing",
        # "Vector Solutions" intact instead of collapsing to the blocklisted "Vector").
        return bool(toks) and " ".join(toks).lower() not in _WHOLE_REJECT

    # Strip title/role/legal/model/date tokens from the ends, but never the last token and
    # never such that the remainder becomes a rejected term.
    while len(tokens) > 1 and (_norm(tokens[0]) in _STRIP_ENDS or _norm(tokens[0]).isdigit()) and _rem_ok(tokens[1:]):
        tokens.pop(0)
    while len(tokens) > 1 and (_norm(tokens[-1]) in _STRIP_ENDS or _norm(tokens[-1]).isdigit()) and _rem_ok(tokens[:-1]):
        tokens.pop()
    if not tokens:
        return None
    cleaned = " ".join(tokens)
    if len(cleaned.replace(" ", "")) <= 1:            # single char like "A"
        return None
    low = cleaned.lower()
    if low in _WHOLE_REJECT:                          # site brand / model / fragment / generic
        return None
    if all(_norm(t) in (_STRIP_ENDS | _WHOLE_REJECT) for t in tokens):
        return None
    return cleaned
_COMPANY_BEFORE_INTERVIEW = re.compile(
    r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,2})\s+[Ii]nterview")


def _company_from_text(text: str) -> Optional[str]:
    for m in _COMPANY_BEFORE_INTERVIEW.finditer(text or ""):
        lead = []
        for w in re.split(r"\s+", m.group(1).strip()):
            if w.lower().strip(".&-") in _NOT_COMPANY:
                break
            lead.append(w)
        if lead:
            return " ".join(lead)
    return None


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)?")
_MD_DANGLING = re.compile(r"\s*\]\([^\s)]*\)?")
_MD_NOISE = re.compile(r"[`*_]+")
# Strips leading list/question markers: bullets, "1.", "1)", "Q1.", "Q1)", "Q1:",
# "Question 1.", "Question 1:" — so question content never starts with a number/label.
_LEAD_MARKER = re.compile(r"^\s*(?:[•\-*]\s*)?(?:(?:q(?:uestion)?\s*)?\d+\s*[.):]\s*)?", re.IGNORECASE)


def _clean_seg(s: str) -> str:
    s = _MD_LINK.sub(r"\1", s)
    s = _MD_DANGLING.sub("", s)
    s = _MD_NOISE.sub("", s)
    s = re.sub(r"^[#>\[\]\-•*\s]+", "", s)
    s = _LEAD_MARKER.sub("", s)  # drop any leading Q1./1)/Question 2: prefix
    s = s.rstrip(" \t`]")
    return re.sub(r"\s+", " ", s).strip()


def _strip_trailing_company(text: str, company: Optional[str]) -> str:
    if not company:
        return text
    m = re.search(r"\s*\(([^)]*)\)\s*$", text)
    if m and m.group(1).strip().lower() == company.strip().lower():
        return text[:m.start()].rstrip()
    return text


_STOPWORDS = {"a", "an", "the"}


def _dedup_key(s: str) -> str:
    """Normalized key for near-duplicate detection: lowercase, drop punctuation
    and leading articles so 'Write a function.' == 'write the function'."""
    tokens = [t for t in re.sub(r"[^\w\s]", "", s.lower()).split() if t not in _STOPWORDS]
    return " ".join(tokens)


def _extract_from_text(text: str) -> List[str]:
    out: List[str] = []
    segments: List[str] = []
    for line in (text or "").splitlines():
        line = _LEAD_MARKER.sub("", line).strip()
        if not line:
            continue
        # Keep a whole line that is already a valid prompt intact (don't shred a
        # multi-sentence "You have 1M users… Design the feed." into fragments);
        # only sentence-split lines that aren't themselves a question.
        if looks_like_question(_clean_seg(line)):
            segments.append(line)
        else:
            segments.extend(s.strip() for s in re.split(r"(?<=[.?!])\s+", line) if s.strip())
    from src.quality import is_quality_question, strip_artifacts
    seen = set()
    for seg in segments:
        seg = strip_artifacts(_clean_seg(seg))
        key = _dedup_key(seg)
        # Form-quality gate (rejects boilerplate/logistics/fragments/headings), not just "looks like a Q".
        if is_quality_question(seg) and key and key not in seen:
            seen.add(key)
            out.append(seg)
            if len(out) >= _PER_RESULT:
                break
    return out


_TERMINAL_PATTERNS = ("usage limit", "exceeds your plan", "forbidden",
                      "unauthorized", "invalid api key", "401", "403")


def _is_terminal_error(errors: list) -> bool:
    """Quota/auth errors won't resolve within a run — worth bailing immediately."""
    joined = " ".join(errors).lower()
    return any(p in joined for p in _TERMINAL_PATTERNS)


def _summarize_tavily_error(errors: list) -> str:
    """Turn raw Tavily exception strings into one human-readable status."""
    joined = " ".join(errors).lower()
    if "usage limit" in joined or "exceeds your plan" in joined or "forbidden" in joined:
        return "Tavily quota/usage limit reached — web search unavailable (upgrade or replace TAVILY_API_KEY)."
    if "unauthorized" in joined or "401" in joined or "invalid api key" in joined:
        return "Tavily API key invalid or unauthorized — web search unavailable."
    if "rate limit" in joined or "429" in joined:
        return "Tavily rate limit hit — web search temporarily unavailable."
    return f"Tavily web search failed: {errors[0]}"


def _search(client, query: str, include_domains: Optional[list] = None,
            errors: Optional[list] = None):
    try:
        kw = dict(query=query, max_results=config.TAVILY_MAX_RESULTS,
                  search_depth="basic", include_raw_content="markdown")
        if include_domains:
            kw["include_domains"] = include_domains
        resp = client.search(**kw)
        return resp.get("results") or []
    except Exception as e:
        # Don't swallow silently — record so callers can surface WHY web search
        # returned nothing (e.g. expired key, quota/usage limit, rate limit).
        msg = f"{type(e).__name__}: {e}"
        print(f"[tavily] search failed for {query!r}: {msg}")
        if errors is not None:
            errors.append(msg)
        return []


def _records_from_results(results, allow: set, seen: set) -> List[Record]:
    out: List[Record] = []
    for r in results:
        url = r.get("url", "") or ""
        dom = domain(url)
        if not url or not _on_allowlist(dom, allow):
            continue
        # Education/tutorial platforms: accept ONLY interview-question pages, not course content
        # (/topics/, /blog/, /article/…). A genuine interview page has "interview" in the URL path.
        if dom in config.EDU_PLATFORM_DOMAINS and "interview" not in url.lower():
            continue
        text = r.get("raw_content") or r.get("content") or ""
        title = r.get("title", "") or ""
        # hellointerview embeds the company in body copy ("... question from Meta"), not the URL.
        hi = _company_from_hellointerview(title + " " + text) if dom.endswith("hellointerview.com") else None
        company = _valid_company(
            _company_from_url(url) or hi or _company_from_text(title) or _company_from_text(text[:200])
        )
        for cand in _extract_from_text(text):
            cand = _strip_trailing_company(cand, company)
            key = _dedup_key(cand)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(Record(question_text=cand, source_url=url, company=company,
                              raw_snippet=(text or "")[:700], source_type=f"tavily:{dom}"))
    return out


# Job-title / role frames so queries surface role-specific interview questions (the session's target
# profiles), not just bare concept queries. Sourced from config TARGET_ROLES so adding a role updates
# both the queries here and the selection bonus. Kept modest (applied to the first term only).
_ROLE_SUFFIXES = list(config.DEFAULT_TARGET_ROLES.get("query_titles", []))


class TavilyConnector:
    name = "tavily"

    def health_check(self) -> tuple:
        """Proactively verify the Tavily API is CALLING CORRECTLY before a full search relies on it.
        Returns (ok: bool, status: str, detail: str). status ∈ ok|no_key|quota|auth|rate|error.
        One lightweight probe call (max_results=1)."""
        if not config.TAVILY_API_KEY:
            return (False, "no_key", "TAVILY_API_KEY not set")
        errors: list = []
        try:
            client = _client()
            resp = client.search(query="generative AI interview questions", max_results=1,
                                  search_depth="basic")
            results = resp.get("results") if isinstance(resp, dict) else None
            if results is None:
                return (False, "error", "Tavily returned an unexpected response shape")
            return (True, "ok", f"{len(results)} result(s)")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            joined = " ".join(errors).lower()
            status = ("quota" if any(p in joined for p in ("usage limit", "exceeds your plan", "forbidden"))
                      else "auth" if any(p in joined for p in ("unauthorized", "invalid api key", "401"))
                      else "rate" if ("rate limit" in joined or "429" in joined)
                      else "error")
            return (False, status, _summarize_tavily_error(errors))

    def fetch(self, outcomes: List[str]) -> tuple:
        """Return (records, search_call_count, error).

        error is None on success, or a short string describing why searches
        failed (e.g. quota/usage limit) so callers can surface it instead of
        silently looking like 'no results found'.

        Searches the FULL INTERVIEW_SOURCE_ALLOWLIST (Tavily caps include_domains ~300; the
        allowlist is well under that) so every trusted source — including blog sources like
        DataCamp that were previously never searched — is reachable. Each concept term gets a
        broad + an attribution-framed query; role/job-title queries are added for the first term
        to keep the call count bounded (~2·terms + roles).
        """
        if not config.TAVILY_API_KEY:
            return [], 0, "TAVILY_API_KEY not set"
        allow = set(config.INTERVIEW_SOURCE_ALLOWLIST or [])
        domains = list(config.INTERVIEW_SOURCE_ALLOWLIST or []) or None
        client = _client()
        records: List[Record] = []
        seen: set = set()
        errors: list = []
        search_count = 0

        terms = [o.replace("_", " ").strip() for o in (outcomes or []) if o and o.strip()][:_MAX_OUTCOMES]

        def run(query: str) -> bool:
            """Run one search; return True if the run should stop (limit / terminal error)."""
            nonlocal search_count
            records.extend(_records_from_results(
                _search(client, query, include_domains=domains, errors=errors), allow, seen))
            search_count += 1
            return len(records) >= _MAX_RECORDS or _is_terminal_error(errors)

        for i, q in enumerate(terms):
            if run(f"{q} interview questions"):
                break
            if run(f"{q} interview question asked at company"):
                break
            # Role/job-title framing only on the first (primary) term — bounds total calls.
            if i == 0:
                stop = False
                for role in _ROLE_SUFFIXES:
                    if run(f"{q} {role} interview questions"):
                        stop = True
                        break
                if stop:
                    break

        error = None
        if not records and errors:
            error = _summarize_tavily_error(errors)
        return records[:_MAX_RECORDS], search_count, error


def _open_web_page_ok(url: str, title: str) -> bool:
    """Is this an interview-question PAGE, on a domain worth reading?

    The single most useful gate for open-web results, and it is measured: across 71 candidates from
    unrestricted searches, ALL 71 passed the form gate while only 29 sat on a page whose URL or title
    said "interview question". The other 42 were forum chatter ("Are you willing to work on this?"),
    GitHub template prose, vendor documentation ("What it does?") and tutorial headings ("What Is n8n?").
    The form gate cannot tell documentation from a question; this can.
    """
    dom = domain(url)
    if not dom:
        return False
    if dom in config.OPEN_WEB_BLOCKED_DOMAINS:
        return False
    if any(dom.startswith(p) for p in config.OPEN_WEB_BLOCKED_PREFIXES):
        return False
    # Also block the blocked domains reached via a subdomain (m.facebook.com, old.reddit.com).
    if any(dom == b or dom.endswith("." + b) for b in config.OPEN_WEB_BLOCKED_DOMAINS):
        return False
    hay = f"{url} {title}".lower()
    return any(tell in hay for tell in config.OPEN_WEB_PAGE_TELLS)


def fetch_open_web(terms: List[str]) -> tuple:
    """LAST-RESORT search with NO domain allowlist. Returns (records, search_calls, error).

    Only called when the bank and the allowlisted web tier have under-delivered for a session — see
    `tools.tool_search_web_questions`. Everything it returns is marked `unvetted_source` upstream and
    never carries a company attribution: an unvetted page is not evidence that a company asked
    something.

    Gates, in order of how much they remove: `_open_web_page_ok` (page + domain), then the shared FORM
    gate. Candidates still face session-fit, the LLM relevance judge and the syllabus audit downstream.
    """
    if not config.TAVILY_API_KEY:
        return [], 0, "TAVILY_API_KEY not set"
    if not config.OPEN_WEB_ENABLED:
        return [], 0, None

    from src.quality import is_quality_question, strip_artifacts

    client = _client()
    records: List[Record] = []
    seen: set = set()
    errors: list = []
    calls = 0
    for term in [t for t in (terms or []) if t and t.strip()][:config.OPEN_WEB_MAX_TERMS]:
        results = _search(client, f"{term} interview questions and answers", include_domains=None)
        calls += 1
        for r in results:
            url = r.get("url", "") or ""
            title = r.get("title", "") or ""
            if not _open_web_page_ok(url, title):
                continue
            text = r.get("raw_content") or r.get("content") or ""
            for cand in _extract_from_text(text):
                cand = strip_artifacts(cand)
                if not is_quality_question(cand):
                    continue
                key = _dedup_key(cand)
                if not key or key in seen:
                    continue
                seen.add(key)
                # company=None ALWAYS: attribution_label turns that into NIAT, and `source_site`
                # records where it actually came from.
                records.append(Record(question_text=cand, source_url=url, company=None,
                                      source_type=f"open_web:{domain(url)}"))
        if len(records) >= config.OPEN_WEB_MAX_RECORDS or _is_terminal_error(errors):
            break
    error = _summarize_tavily_error(errors) if (not records and errors) else None
    return records[:config.OPEN_WEB_MAX_RECORDS], calls, error


def search_question(question: str) -> List[Record]:
    """Research loop helper: find more allowlisted attribution for one confirmed question."""
    if not config.TAVILY_API_KEY:
        return []
    allow = set(config.INTERVIEW_SOURCE_ALLOWLIST or [])
    results = _search(_client(), f'"{question}" interview asked at company',
                      include_domains=_ATTRIBUTION_DOMAINS + _BROAD_DOMAINS)
    return _records_from_results(results, allow, set())
