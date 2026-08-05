"""Form-quality gate tests.

Every case here is a real string observed in data/genai_question_bank.json or in the reviewer
feedback log — not invented examples. The two regression classes that matter:

  * garbage that USED to pass and polluted question sets (blog titles, SEO tails, fragments)
  * good questions that a previous tightening of the filters WRONGLY dropped

The second group is the important one: each was a false positive caught during development, and a
future filter change that re-breaks any of them is a regression even if it removes more garbage.
"""
import pytest

from src.quality import is_quality_question, strip_artifacts

# Garbage observed in the built bank, with the class of defect each represents.
GARBAGE = [
    ("Is GPT Image 2 the Best Image Generation Model?", "clickbait comparison headline"),
    ("Closed Source Model — Locked In?", "editorial headline, title-cased"),
    ("Build Human-Like AI Voice App with Gemini 3.1 Flash TTS", "tutorial/video title"),
    ("Can Voice Agents Handle Bilingual Customers?", "blog title shaped like a question"),
    ("What Are Generative AI Interview Questions?", "listicle article title"),
    ("Top 5 Diffusion Models Explained", "bare heading, no question"),
    ("What happens during generation", "truncated lesson heading"),
    ("how does it affect generation?", "pronoun-subject fragment"),
    ("why is it important?", "pronoun-subject fragment"),
    ("how can they be reduced?", "pronoun-subject fragment"),
    ("How does it differ from a standard autoencoder?", "pronoun-subject fragment"),
    ("If I get a chance, can I move to the AI agents or generative AI stack?", "candidate-voice"),
    ("On a scale of 1 to 10, how accurate is your AI in creating a workflow?",
     "interviewer rating one candidate's own project"),
    ("What to expect from your interview", "page boilerplate"),
    ("How long is the interview process?", "hiring logistics"),
    # Possessive/demonstrative references whose antecedent stayed on the source page. All of these
    # were live in the shipped bank and reached a real generated question set before being caught.
    ("What are its key components?", "possessive-reference fragment"),
    ("how does its architecture work?", "possessive-reference fragment"),
    ("what are their key use cases?", "possessive-reference fragment"),
    ("Explain the role of the retrieval component in these systems.",
     "demonstrative-reference fragment"),
    ("What are the trade-offs in such scenarios?", "demonstrative-reference fragment"),
    ("Describe the process as shown above.", "reference to page content"),
]

# Real questions that MUST survive. Several are reviewer-approved; the rest were false positives
# found while tuning the title-case heuristics.
LEGITIMATE = [
    "What are the core components of an AI Agent?",
    "Why is memory critical for the performance of AI agents?",
    "What are Prompt Engineering Techniques?",       # 80% title-cased, reviewer-APPROVED
    "How do LLMs work?",                            # short, non-definitional
    "What is RAG?",                                 # very short, definitional
    "What is a prompt?",
    "Explain Stable diffusion",
    "Explain Mode Collapse in GANs.",               # 80% title-cased, technical
    "Explain GANs (Generative Adversarial Networks)",   # parenthetical acronym gloss
    "Design the GPU Job Scheduler for a Text-to-Video Generation Service",
    "Design an end-to-end image generation system. Cover the following:",
    "How can I reduce inference latency in a production LLM app?",   # first-person but technical
    "Compare LoRA and full fine-tuning",
    "How do agents decide when to stop a task?",
    "Implement a retry loop for malformed tool-call JSON",
    "Fine-Tuning vs RAG - which one should you use?",   # internal dash, not an SEO tail
    "How do you prevent hallucinations?",
    # Named antecedents — these must survive the reference-fragment rules above.
    "What are the key components of an AI agent?",
    "Explain the role of the retrieval component in a RAG pipeline.",
    "What are the main advantages of LoRA over full fine-tuning?",
    "Compare these two approaches: RAG and fine-tuning.",
]


@pytest.mark.parametrize("text,defect", GARBAGE, ids=[d for _, d in GARBAGE])
def test_rejects_garbage(text, defect):
    assert not is_quality_question(text), f"should reject ({defect}): {text!r}"


@pytest.mark.parametrize("text", LEGITIMATE)
def test_keeps_real_questions(text):
    assert is_quality_question(text), f"should keep: {text!r}"


class TestStripArtifacts:
    def test_removes_seo_site_tail(self):
        assert strip_artifacts(
            "Build an Enterprise RAG Workflow | Dataford Interview Questions"
        ) == "Build an Enterprise RAG Workflow"

    def test_keeps_meaningful_internal_dash(self):
        """A dash inside the question is content, not site furniture."""
        assert "which one should you use" in strip_artifacts(
            "Fine-Tuning vs RAG - which one should you use?"
        )

    def test_removes_scrape_markers(self):
        assert strip_artifacts("Q: What is RAG?") == "What is RAG?"
        assert strip_artifacts("\\\\What is RAG?") == "What is RAG?"

    def test_preserves_hyphenated_terms(self):
        """"Q-learning" must not be mistaken for a "Q." scrape marker and truncated."""
        assert strip_artifacts("What is Q-learning?") == "What is Q-learning?"

    def test_restores_question_mark_on_wh_heading(self):
        assert strip_artifacts("What is gradient descent").endswith("?")

    def test_empty_input(self):
        assert strip_artifacts("") == ""
        assert not is_quality_question("")
        assert not is_quality_question(None)
