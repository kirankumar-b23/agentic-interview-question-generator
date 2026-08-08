"""Attribution and company-validation tests.

The product promise is that a question is labelled with a REAL company in uppercase or the honest
`NIAT` placeholder — never a website brand, a page-topic word, or a fabricated employer. These
tests pin that promise, including the specific artifacts found in the built bank.
"""
import pytest

from src.models import QuestionDetail, attribution_label
from src.sources.tavily_search import _valid_company


class TestValidCompany:
    # Bare-word artifacts found in data/genai_question_bank.json. "Tech" came from
    # techinterviewhandbook.org (a source site); the others are page-topic words.
    @pytest.mark.parametrize("junk", ["Tech", "Product", "Classification", "Ensemble",
                                      "GPT", "Crack", "A", "", "unknown", "confidential"])
    def test_rejects_non_companies(self, junk):
        assert _valid_company(junk) is None, f"{junk!r} is not a company"

    # Real employers seen in the placement data / harvested pages. Unfamiliar names are still real.
    @pytest.mark.parametrize("name", ["Anthropic", "OpenAI", "Capgemini", "Hugging Face",
                                      "Genentech", "Mercor", "Palnesto", "NextBill",
                                      "RedFerns Tech", "Ignitetech"])
    def test_keeps_real_companies(self, name):
        assert _valid_company(name) == name

    def test_none_input(self):
        assert _valid_company(None) is None

    def test_multiword_name_containing_a_rejected_word_survives(self):
        """"Tech" alone is junk, but it must not be stripped out of a real multi-word name."""
        assert _valid_company("RedFerns Tech") == "RedFerns Tech"


class TestAttributionLabel:
    def test_real_company_is_uppercased(self):
        assert attribution_label("Anthropic", "interview_db", None) == "ANTHROPIC"

    def test_no_company_and_no_url_uses_placeholder(self):
        assert attribution_label(None, "interview_db", None) == "NIAT"
        assert attribution_label("   ", "interview_db", None) == "NIAT"

    def test_no_company_does_not_borrow_the_site_name(self):
        """Attribution answers WHO ASKED, so it is company → NIAT with no site fallback.

        It used to fall back to the source site, and a real run shipped questions attributed to
        "Analytics Vidhya", "Indeed", "Edureka" and "DataCamp" in the same field as a genuine
        "ANTHROPIC" — one of those is an employer and the rest are content sites. Provenance is not
        lost; it moved to `QuestionDetail.source_site` (see the next test)."""
        assert attribution_label(None, "web", "https://www.geeksforgeeks.org/llm-interview") == "NIAT"
        assert attribution_label(None, "web", "https://www.indeed.com/hire/job-description/x") == "NIAT"

    def test_provenance_is_still_available_separately(self):
        """The site name survives as provenance on its own field, rendered as a brand not a domain."""
        from src.models import QuestionDetail

        q = QuestionDetail(category="GEN_AI", content="What is RAG?", topic="Gen AI", source="web",
                           source_url="https://www.geeksforgeeks.org/llm-interview")
        assert q.source_site == "GeeksforGeeks"
        assert ".org" not in q.source_site and "www." not in q.source_site
        assert q.attribution == "NIAT"      # the two are never the same field

    def test_company_wins_over_source_url(self):
        """A known employer must never be replaced by the site the question was found on."""
        assert attribution_label(
            "Anthropic", "web", "https://www.geeksforgeeks.org/llm-interview"
        ) == "ANTHROPIC"


class TestQuestionDetail:
    def _q(self, **kw):
        base = dict(category="GEN_AI", content="What is RAG?", topic="Gen AI", source="interview_db")
        base.update(kw)
        return QuestionDetail(**base)

    def test_attribution_is_computed_from_company(self):
        assert self._q(asked_in_company="Capgemini").attribution == "CAPGEMINI"

    def test_attribution_without_company(self):
        assert self._q().attribution == "NIAT"

    def test_new_scoring_fields_default_to_none(self):
        """Absent scores must be None, not 0.0 — a missing measurement is not a zero one."""
        q = self._q()
        assert q.session_fit is None
        assert q.retrieval_score is None
        assert q.relevance_score is None

    def test_scoring_fields_round_trip(self):
        q = self._q(session_fit=0.42, retrieval_score=0.31)
        assert QuestionDetail.model_validate(q.model_dump()).session_fit == 0.42

    def test_list_expected_answer_is_coerced_to_text(self):
        assert self._q(expected_answer=["one", "two"]).expected_answer == "one\ntwo"
