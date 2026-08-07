"""Tool-name queries, and the last-resort open-web tier.

Both come from probing the n8n gap with the real Tavily API. What was measured:

* `"n8n workflow automation interview questions"` → **0** allowlisted records, 24 open-web candidates.
* `"n8n node types interview questions"` → **14** records from the SAME 67-domain allowlist.
  So the trusted sources had n8n content; nothing ever asked them for it, because queries are built from
  `interview_topics` ("Workflow automation with LLMs", "Chaining LLM calls in workflows") and the string
  "n8n" never entered one.
* Of **71** open-web candidates, **all 71 passed the form gate** but only **29** sat on a genuine
  interview-question page. The other 42 were forum chatter, GitHub template prose, vendor docs and
  tutorial headings — so an open tier needs a page-level filter, not just the form gate.

No network here: the page filter and term extraction are pure, and the tier is driven with stubs.
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail


# ── Phase 1: ask for the tools by name ──────────────────────────────────────

class TestToolTermExtraction:
    def test_it_finds_the_tool_that_was_never_queried(self):
        from src.tools import _tool_terms

        ctx = SimpleNamespace(
            scope_in=["n8n node types (Chat Trigger, Google Sheets, Slack, HTTP Request, SplitInBatches)",
                      "Connections between nodes in n8n"],
            key_concepts=["n8n platform introduction"], learning_outcomes=[])
        assert "n8n" in _tool_terms(ctx)

    @pytest.mark.parametrize("text,expected", [
        ("Configure and authenticate Google Gemini API for content generation", "Google Gemini"),
        ("Set up and run Stable Diffusion locally using cloud GPU resources", "Stable Diffusion"),
        ("Implement web research integration (SerpAPI) within an AI agent workflow", "SerpAPI"),
        ("Set up a Kaggle account with phone verification to enable GPU access", "Kaggle"),
    ])
    def test_it_extracts_real_product_names(self, text, expected):
        from src.tools import _tool_terms

        ctx = SimpleNamespace(scope_in=[], key_concepts=[], learning_outcomes=[text])
        assert expected in _tool_terms(ctx)

    def test_a_leading_verb_is_not_part_of_the_product_name(self):
        """"Manage Kaggle session duration…" must yield "Kaggle", not "Manage Kaggle" — the query
        "Manage Kaggle interview questions" is a wasted Tavily call."""
        from src.tools import _tool_terms

        ctx = SimpleNamespace(scope_in=[], key_concepts=[],
                              learning_outcomes=["Manage Kaggle session duration and GPU quota"])
        terms = _tool_terms(ctx)
        assert "Kaggle" in terms
        assert not any(t.lower().startswith("manage") for t in terms)

    def test_a_theory_session_yields_no_tools_rather_than_noise(self):
        """Advanced Prompt Engineering teaches no product. Inventing a term would waste a search."""
        from src.tools import _tool_terms

        ctx = SimpleNamespace(scope_in=[], key_concepts=[], learning_outcomes=[
            "Master zero-shot, one-shot, few-shot, and chain-of-thought prompting techniques",
            "Decompose complex problems into step-by-step reasoning prompts"])
        assert _tool_terms(ctx) == []

    def test_no_context_is_empty_not_an_error(self):
        from src.tools import _tool_terms

        assert _tool_terms(None) == []


# ── Phase 2: which PAGES the open tier will read ────────────────────────────

class TestOpenWebPageFilter:
    """The 29-vs-42 discriminator, using the exact domains the live probe returned."""

    @pytest.mark.parametrize("url,title", [
        ("https://growai.in/n8n-interview-questions/", "N8N Interview Questions"),
        ("https://www.interviewkickstart.com/interview-questions/n8n", "n8n questions"),
        ("https://vskills.in/practice/n8n-interview-questions", "n8n interview questions"),
    ])
    def test_genuine_interview_question_pages_are_read(self, url, title):
        from src.sources.tavily_search import _open_web_page_ok

        assert _open_web_page_ok(url, title) is True

    @pytest.mark.parametrize("url,title,why", [
        ("https://community.n8n.io/t/help-with-nodes/123", "Help with nodes", "forum chatter"),
        ("https://ai.google.dev/gemini-api/docs", "Gemini API docs", "vendor documentation"),
        ("https://github.com/x/n8n-templates", "n8n templates", "code host, template prose"),
        ("https://www.facebook.com/groups/n8n", "n8n group", "social"),
        ("https://m.facebook.com/groups/n8n", "n8n group", "social via subdomain"),
        ("https://www.linkedin.com/posts/n8n-tips", "n8n tips", "social"),
        ("https://medium.com/@a/what-is-n8n", "What Is n8n?", "tutorial heading"),
        ("https://stable-diffusion-art.com/automatic1111/", "AUTOMATIC1111 tutorial", "tutorial"),
        ("https://old.reddit.com/r/n8n/comments/abc", "n8n thread", "forum"),
    ])
    def test_the_noisy_sources_are_refused(self, url, title, why):
        from src.sources.tavily_search import _open_web_page_ok

        assert _open_web_page_ok(url, title) is False, why

    def test_a_missing_url_is_refused_not_crashed(self):
        from src.sources.tavily_search import _open_web_page_ok

        assert _open_web_page_ok("", "interview questions") is False


# ── Phase 2: the tier itself ────────────────────────────────────────────────

def _result(url, title, body):
    return {"url": url, "title": title, "raw_content": body}


class TestOpenWebTier:
    # Verbatim from the live probe: the clean ones and the junk, in one page body each.
    CLEAN = ("What are nodes in N8N and how are they categorized?\n"
             "What is the HTTP Request node and when do you use it?\n"
             "What is the difference between IF node and Switch node?\n"
             "What is the Merge node and what merge modes does it support?\n")
    JUNK = ("Are you willing to work on this?\nAny resources to support this?\nWhat Is n8n?\n")

    def test_it_keeps_the_clean_questions_and_never_claims_a_company(self, monkeypatch):
        import src.sources.tavily_search as tv

        monkeypatch.setattr(tv, "_client", lambda: object())
        monkeypatch.setattr(tv, "_search", lambda *a, **k: [
            _result("https://growai.in/n8n-interview-questions/", "N8N Interview Questions", self.CLEAN)])

        records, calls, err = tv.fetch_open_web(["n8n"])
        assert err is None and calls == 1
        assert len(records) >= 3
        assert all(r.company is None for r in records), "an unvetted page cannot assert a company"
        assert all(r.source_type.startswith("open_web:") for r in records)
        assert any("Merge node" in r.question_text for r in records)

    def test_a_blocked_page_contributes_nothing(self, monkeypatch):
        import src.sources.tavily_search as tv

        monkeypatch.setattr(tv, "_client", lambda: object())
        monkeypatch.setattr(tv, "_search", lambda *a, **k: [
            _result("https://community.n8n.io/t/x/1", "Help", self.CLEAN)])   # clean text, bad page
        records, _, _ = tv.fetch_open_web(["n8n"])
        assert records == [], "the page filter must apply regardless of how good the text looks"

    def test_the_form_gate_alone_does_NOT_catch_a_short_tutorial_heading(self, monkeypatch):
        """Documents a real limitation rather than pretending it away.

        `is_quality_question("What Is n8n?")` returns True: it ends in "?" and is too short (3 words)
        for the article-title heuristic, which needs >=5 so genuine short asks like "What is RAG?"
        survive. Title case cannot separate "What Is n8n?" from "What is RAG?" at that length.

        So the page filter is what keeps such text out — this asserts that the tier does not somehow
        rescue it, and that the residual risk is carried by session-fit, the relevance judge and the
        syllabus audit downstream, not by the form gate.
        """
        from src.quality import is_quality_question
        import src.sources.tavily_search as tv

        assert is_quality_question("What Is n8n?") is True, "the limitation is real; do not assert away"

        monkeypatch.setattr(tv, "_client", lambda: object())
        monkeypatch.setattr(tv, "_search", lambda *a, **k: [
            _result("https://growai.in/n8n-interview-questions/", "N8N Interview Questions", self.JUNK)])
        records, _, _ = tv.fetch_open_web(["n8n"])
        # Forum chatter IS filtered — it ends on a bare demonstrative with nothing named for the
        # pronoun to refer to, a rule added because these exact lines passed every other gate.
        texts = [r.question_text for r in records]
        assert "Are you willing to work on this?" not in texts
        assert "Any resources to support this?" not in texts
        # …while a self-contained trailing pronoun survives, because the question names its subject.
        assert is_quality_question("What is the HTTP Request node and when do you use it?") is True

    def test_disabled_by_config_means_no_calls(self, monkeypatch):
        import src.config as cfg
        import src.sources.tavily_search as tv

        monkeypatch.setattr(cfg, "OPEN_WEB_ENABLED", False)
        called = []
        monkeypatch.setattr(tv, "_search", lambda *a, **k: called.append(1) or [])
        records, calls, _ = tv.fetch_open_web(["n8n"])
        assert records == [] and calls == 0 and not called

    def test_no_api_key_is_reported_not_crashed(self, monkeypatch):
        import src.config as cfg
        import src.sources.tavily_search as tv

        monkeypatch.setattr(cfg, "TAVILY_API_KEY", "")
        records, calls, err = tv.fetch_open_web(["n8n"])
        assert records == [] and calls == 0 and "TAVILY_API_KEY" in err


# ── Provenance of what it produces ──────────────────────────────────────────

class TestUnvettedProvenance:
    def test_an_unvetted_question_reads_NIAT_and_keeps_its_site(self):
        q = QuestionDetail(category="GEN_AI", content="What are nodes in N8N?", topic="Gen AI",
                           source="web", source_url="https://growai.in/n8n-interview-questions/",
                           unvetted_source=True)
        assert q.attribution == "NIAT", "never imply a company asked this"
        assert q.source_site == "Growai"
        assert q.unvetted_source is True

    def test_the_flag_defaults_off_so_allowlisted_questions_are_unaffected(self):
        q = QuestionDetail(category="GEN_AI", content="What is chain-of-thought prompting?",
                           topic="Gen AI", source="web",
                           source_url="https://www.geeksforgeeks.org/x")
        assert q.unvetted_source is False

    def test_the_report_names_the_count(self):
        """A set that reached outside the trusted list must say so, with numbers."""
        from src.agent import AgentState
        from src.data_loader import get_data_store
        from src.pipeline import _build_quality_report

        st = AgentState(config=GenerationConfig(session_names=["S"], max_questions=15),
                        data_store=get_data_store())
        st.session_context = SimpleNamespace(
            learning_outcomes=["Understand n8n nodes"], interview_topics=["n8n node types"],
            key_concepts=[], scope_in=[], scope_out=[], matched_kp_ids=[], session_type="mixed")
        for i in range(5):
            q = QuestionDetail(category="GEN_AI", content=f"What is the node {i}?", topic="Gen AI",
                               source="web", unvetted_source=(i < 2), session_fit=0.7,
                               relevance_score=0.8)
            st.questions[q.question_id] = q
        rep = _build_quality_report(st, 0)
        assert any("outside the trusted source list" in n.lower() for n in rep.critique)
        assert any("2 of 5" in n for n in rep.critique)


class TestTheTriggerThatNeverFired:
    """When the tier engages. This is the whole reason it had never run on a live run.

    Run 8fb9fcb3 requested 15, survived validation with EXACTLY 5, and the guard
    `surviving >= MIN_QUESTIONS` (MIN_QUESTIONS == 5) read that as satisfied — so the tier written for
    the n8n gap was skipped, and the shipped set contained 0 questions about n8n, RSS nodes, Schedule
    Trigger or Gmail Send Node while 10 of the topic's 22 outcomes named them.

    The table is asserted rather than the constant, because the off-by-one is the defect: a ratio ALONE
    reintroduces it whenever `requested == MIN_QUESTIONS` (0.6 * 5 floors to 5, and `surviving < 5` is
    false at exactly 5 all over again).
    """

    # (surviving, requested, should_fire, why)
    TABLE = [
        (4, 15, True, "below any threshold"),
        (5, 15, True, "THE 8fb9fcb3 CASE — exactly MIN_QUESTIONS against a request of 15"),
        (6, 15, True, "still materially short of 15 (threshold 9)"),
        (7, 15, True, "still short — a 7/15 set is what 'always 5' looks like"),
        (9, 15, False, "reached the 60% threshold"),
        (12, 15, False, "close enough to the ask that open-web noise is not worth it"),
        (4, 5, True, "under the floor"),
        (5, 5, True, "at the floor with a request of 5 — the ratio alone would skip this"),
        (6, 5, False, "above the floor and above the ask"),
        (12, 5, False, "well above the ask"),
    ]

    @pytest.mark.parametrize("surviving,requested,should_fire,why", TABLE)
    def test_the_trigger_table(self, surviving, requested, should_fire, why):
        from src.pipeline import _open_web_shortfall

        fired = _open_web_shortfall(surviving, requested) is not None
        assert fired is should_fire, f"surviving={surviving} requested={requested}: {why}"

    def test_a_full_set_that_ignores_the_session_tools_still_fires(self):
        """The count trigger cannot see a coverage gap. 15 questions on an n8n session that never say
        "n8n" is still the wrong set."""
        from src.pipeline import _open_web_shortfall

        assert _open_web_shortfall(15, 15) is None
        reason = _open_web_shortfall(15, 15, missing_tools=["n8n", "Gmail Send Node"])
        assert reason is not None and "n8n" in reason

    def test_the_reason_names_the_numbers_so_a_live_run_is_debuggable(self):
        """The old version emitted nothing distinguishing, which is why a dead tier went unnoticed."""
        from src.pipeline import _open_web_shortfall

        reason = _open_web_shortfall(5, 15)
        assert "5" in reason and "15" in reason


class TestUnrepresentedTerms:
    """Which taught tools nothing asks about — the A2 trigger's input."""

    @staticmethod
    def _qs(*contents):
        return [SimpleNamespace(content=c) for c in contents]

    def test_it_finds_the_tool_no_question_mentions(self):
        from src.pipeline import _unrepresented_terms

        missing = _unrepresented_terms(
            ["n8n", "Gemini"],
            self._qs("How do you write effective prompts for consistent JSON output?",
                     "Design a system prompt for a Gemini-powered assistant."))
        assert missing == ["n8n"]

    def test_a_represented_tool_is_not_reported(self):
        from src.pipeline import _unrepresented_terms

        assert _unrepresented_terms(
            ["n8n"], self._qs("What is the Merge node in n8n and what modes does it support?")) == []

    def test_it_matches_on_a_word_boundary_not_a_substring(self):
        """"RSS" must not be satisfied by "across", nor "n8n" by a longer token."""
        from src.pipeline import _unrepresented_terms

        assert _unrepresented_terms(["RSS"], self._qs("How do you scale across regions?")) == ["RSS"]

    def test_no_terms_means_nothing_missing_rather_than_everything(self):
        """A theory-only session yields no tool terms, and must not trigger the tier."""
        from src.pipeline import _unrepresented_terms

        assert _unrepresented_terms([], self._qs("What is RAG?")) == []


def _ctx_teaching_n8n():
    return SimpleNamespace(
        scope_in=["n8n node types (Chat Trigger, Google Sheets, HTTP Request)"],
        key_concepts=["Workflow automation with n8n"],
        learning_outcomes=["Configure RSS Feed Read Node in n8n to fetch articles"],
        interview_topics=["Workflow automation with LLMs"],
        matched_kp_ids=[], scope_out=[], session_type="code_heavy")


def _state_with(questions, ctx=None):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=["Build Your Own AI News Summarizer"]),
                    data_store=get_data_store())
    st.session_context = ctx or _ctx_teaching_n8n()
    st.questions = {q.question_id: q for q in questions}
    return st


def _q(content, **kw):
    return QuestionDetail(question_id=kw.pop("qid", None) or content[:24],
                          category="GEN_AI", content=content, topic="t",
                          difficulty=kw.pop("difficulty", "Medium"), source=kw.pop("source", "web"),
                          **kw)


class TestToolQuestionsSurviveTheCrossTopicPrefilter:
    """A3. Run 8fb9fcb3 retrieved exactly ONE real n8n question and this stage dropped it.

    `_prefilter_semantic` compares a candidate against POOLED course-topic profiles, and pooled across
    the whole GenAI course an n8n question resembles "the course" less than a prompt-engineering
    question does. That is structural, so it keeps happening — hence the exemption for a question that
    NAMES a tool this session teaches.
    """

    N8N = "What kind of workflows have you built with n8n before, and what broke?"
    OTHER = "How do you tune a diffusion model's guidance scale for photorealism?"

    def _run(self, monkeypatch):
        """Force the worst case: every candidate looks like it belongs to another topic."""
        import src.pipeline as pl
        from src import embeddings

        monkeypatch.setattr(pl, "_topic_profiles",
                            lambda names: (["News Summarizer"], ["Image Generation prose"]))

        def fake_matrix(contents, other=None):
            # cur_profile is passed first, other_texts second; make "belongs elsewhere" true for all.
            hi = other is not None and other == ["Image Generation prose"]
            return [[0.9 if hi else 0.05] for _ in contents]

        monkeypatch.setattr(embeddings, "cosine_matrix", fake_matrix)
        st = _state_with([_q(self.N8N), _q(self.OTHER)])
        pl.AgentPipeline()._prefilter_semantic(st, lambda *a, **k: None)
        return st

    def test_the_n8n_question_is_kept(self, monkeypatch):
        st = self._run(monkeypatch)
        kept = [q.content for q in st.questions.values()]
        assert self.N8N in kept, "a question naming a tool this session teaches must not be pre-filtered"

    def test_a_genuinely_off_topic_question_is_still_dropped(self, monkeypatch):
        """The exemption must not become a blanket pass — that would defeat the stage."""
        st = self._run(monkeypatch)
        kept = [q.content for q in st.questions.values()]
        assert self.OTHER not in kept
        assert any(r["stage"] == "off_topic_prefilter" for r in st.removed)


class TestOpenWebAdditionsAreScored:
    """A4. Firing the tier lit up a path that had never run live.

    `add_open_web_records` does not score session fit, so additions landed with `session_fit = None`.
    `_build_quality_report` averages only non-None fits, so `session_grounding` would have measured
    just the VETTED subset while reporting a whole-set number; and `_rank_key` reads None as 0.0, so
    every unvetted addition sank to the bottom of Review's fit ranking regardless of quality.
    """

    @staticmethod
    def _scored_state(monkeypatch, n_vetted=35):
        import src.pipeline as pl
        from src import embeddings

        monkeypatch.setattr(pl, "_session_profile", lambda names, ctx: (["n8n workflows"], []))
        vetted = [_q(f"Vetted question number {i} about prompts?", qid=f"v{i}") for i in range(n_vetted)]
        new = [_q("What is the Merge node in n8n?", qid="new-good"),
               _q("What is the capital of France?", qid="new-bad")]
        st = _state_with(vetted + new)
        for q in vetted:                       # already decided by the earlier full-pool pass
            q.session_fit = 0.20

        def fake_matrix(contents, profile=None):
            return [[0.90 if "Merge node" in c else 0.01] for c in contents]

        monkeypatch.setattr(embeddings, "cosine_matrix", fake_matrix)
        pl.AgentPipeline()._score_session_fit(st, lambda *a, **k: None,
                                              only_ids={"new-good", "new-bad"})
        return st

    def test_every_surviving_addition_carries_a_real_session_fit(self, monkeypatch):
        st = self._scored_state(monkeypatch)
        assert st.questions["new-good"].session_fit is not None
        assert st.questions["new-good"].session_fit > 0.5

    def test_a_weak_addition_is_dropped_on_the_session_bar(self, monkeypatch):
        st = self._scored_state(monkeypatch)
        assert "new-bad" not in st.questions
        assert any(r["stage"] == "session_fit" for r in st.removed)

    def test_it_never_re_decides_the_vetted_set(self, monkeypatch):
        """The reason this takes `only_ids` at all: the floor is relative to the pool's best fit, so a
        blanket re-run after adding a high-scoring batch would raise the bar and evict questions whose
        place was already settled. Here every vetted fit (0.20) sits under the new floor
        (0.5 * 0.90 = 0.45) and they must all survive anyway."""
        st = self._scored_state(monkeypatch)
        assert sum(1 for qid in st.questions if qid.startswith("v")) == 35
        assert all(r.get("content", "").find("Vetted question") == -1 for r in st.removed)


class TestPlatformQualifiedQueries:
    """The terms are single tokens, and bare they retrieve the wrong universe.

    A live run searched "RSS, n8n, Merge" — the tier fired correctly, and every candidate it brought
    back was about Merge the COMPANY, `pandas.merge()`, merging sorted lists, or ServiceNow's RSS web
    service. The relevance judge rejected all of them, so the tier spent its calls and added nothing.
    The cause is upstream of both: the reading material writes "Merge and Aggregate nodes" and "RSS Feed
    Read Node" with a lowercase head, so `_PROPER_RUN` can only ever capture "Merge" / "RSS".
    """

    def test_ambiguous_terms_are_qualified_with_the_platform(self):
        from src.tools import _qualify_tool_terms

        assert _qualify_tool_terms(["n8n", "Merge", "RSS"]) == ["n8n", "n8n Merge", "n8n RSS"]

    def test_the_platform_itself_is_not_doubled(self):
        from src.tools import _qualify_tool_terms

        assert "n8n n8n" not in _qualify_tool_terms(["n8n", "Merge"])

    def test_a_term_already_naming_the_platform_is_left_alone(self):
        from src.tools import _qualify_tool_terms

        assert _qualify_tool_terms(["n8n", "n8n nodes"]) == ["n8n", "n8n nodes"]

    def test_with_no_platform_token_the_terms_are_unchanged(self):
        """A theory session must not have its terms mangled by a guessed qualifier."""
        from src.tools import _qualify_tool_terms

        assert _qualify_tool_terms(["Gemini", "Merge"]) == ["Gemini", "Merge"]

    def test_blank_terms_are_dropped_not_prefixed(self):
        from src.tools import _qualify_tool_terms

        assert _qualify_tool_terms(["n8n", "", "  "]) == ["n8n"]

    def test_the_platform_is_the_highest_ranked_identity_token(self):
        """`_tool_terms` is ordered most-mentioned-first, so the first letter-digit token wins."""
        from src.tools import _qualify_tool_terms

        assert _qualify_tool_terms(["Automatic1111", "LoRA"]) == ["Automatic1111", "Automatic1111 LoRA"]


class TestToolTermsAgainstRealCuratedData:
    """`_tool_terms` feeds a LIVE Tavily search, so a concept leaking in costs quota.

    `TestToolTermExtraction` above uses crafted inputs and passed while the real curated outcomes
    produced `['Acting', 'Reasoning', 'Observation', 'ReAct']` for an AI-agents session. Because
    `_unrepresented_terms` fires the open-web tier for any taught tool no question covers, a live run
    fanned out asking Tavily for "Observation interview questions and answers" and exhausted the plan's
    usage limit. These cases read the shipped `session_outcomes.review.json` instead.
    """

    @staticmethod
    def _curated():
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "data/reading_materials/session_outcomes.review.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _terms_for(self, needle):
        from types import SimpleNamespace
        from src.tools import _tool_terms

        for name, v in self._curated().items():
            if needle.lower() in name.lower():
                return name, _tool_terms(SimpleNamespace(
                    scope_in=v.get("scope_in") or [], key_concepts=v.get("key_concepts") or [],
                    learning_outcomes=v.get("learning_outcomes") or [], interview_topics=[]))
        return None, None

    # Words this curriculum capitalises mid-sentence that must never reach a search query.
    CONCEPTS = {"acting", "reasoning", "observation", "thought", "action", "reflection", "memory",
                "planning", "role", "tone", "format", "write", "read", "prepare", "handle", "provide"}

    def test_no_curated_session_yields_a_concept_as_a_tool(self):
        from types import SimpleNamespace
        from src.tools import _tool_terms

        curated = self._curated()
        if not curated:
            pytest.skip("curated outcomes not present")
        leaks = {}
        for name, v in curated.items():
            terms = _tool_terms(SimpleNamespace(
                scope_in=v.get("scope_in") or [], key_concepts=v.get("key_concepts") or [],
                learning_outcomes=v.get("learning_outcomes") or [], interview_topics=[]))
            bad = [t for t in terms if t.lower() in self.CONCEPTS]
            if bad:
                leaks[name] = bad
        assert not leaks, f"concepts would be queried as tools: {leaks}"

    def test_the_theory_session_that_burned_the_tavily_quota(self):
        name, terms = self._terms_for("Introduction to AI Agents")
        if name is None:
            pytest.skip("session not in curated outcomes")
        assert "Observation" not in terms and "Reasoning" not in terms and "Acting" not in terms
        assert terms == ["ReAct"], f"expected only the framework name, got {terms}"

    def test_the_tool_session_keeps_its_products(self):
        """The other half of the trade: pruning concepts must not cost real product names, which is why
        this is a denylist and not a whitelist of product shapes."""
        name, terms = self._terms_for("News Summarizer | Part 1")
        if name is None:
            pytest.skip("session not in curated outcomes")
        assert "n8n" in terms and "RSS" in terms
