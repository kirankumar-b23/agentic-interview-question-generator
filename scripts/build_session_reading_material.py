"""Build a precise per-session reading-material map.

Produces data/reading_materials/session_map.json, keyed by the CANONICAL session
names in data/course_structure.json, with each value being only that session's
reading-material text.

Why this exists
---------------
The two course reading-material files are monolithic. Their headings do NOT match
the canonical session names (e.g. "# Introduction" vs "Exploring Gen AI
Capabilities"), so the old fuzzy substring matching in data_loader fed the wrong
content into understand_session — the root cause of off-topic questions.

Two different layouts, handled explicitly:

  gen_ai_reading_material.md
    Each session now starts with its own `# <Title>` heading. We anchor on those
    exact headings (GEN_AI_ANCHORS below), slicing each session from its anchor to
    the next in file order. Two sessions still sit under a generic `# Introduction`
    heading, so their anchor bridges to a unique first-paragraph phrase.

  llm_applications_reading_material.md
    No quote boundaries. Sessions start at distinctive `# <Heading>` lines that DO
    match canonical names. We anchor on those exact headings (LLM_APPS_ANCHORS),
    which also avoids the code-comment false-headings (`# Getting weather data...`)
    that broke the old generic parser.

Sessions with no clean reading material (a few bleed/merge in the source files)
simply get no entry; understand_session then falls back to the knowledge graph.

Run: python scripts/build_session_reading_material.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GEN_AI_RM, LLM_APPS_RM, DATA_DIR  # noqa: E402

OUT_PATH = DATA_DIR / "reading_materials" / "session_map.json"
MAX_CHARS = 12000  # generous per-session cap

# gen_ai: canonical session name -> regex anchoring its `# <Title>` heading line.
# Most are exact title headings; the two generic `# Introduction` sessions (Kaggle
# setup, F5-TTS audio) bridge to a unique first-paragraph phrase so they resolve to
# the right occurrence. Order here is display-only — slicing is by file position.
GEN_AI_ANCHORS = [
    ("Generative AI Foundations", r'^#\s+Generative AI Foundations\s*$'),
    ("Exploring Gen AI Capabilities", r'^#\s+Exploring Gen AI Capabilities\s*$'),
    ("Productivity Power Up with AI Tools", r'^#\s+Productivity Power Up with AI Tools\s*$'),
    ("Prompt Engineering Fundamentals", r'^#\s+Prompt Engineering Fundamentals\s*$'),
    ("Building Social Media Content Automation Workflow | Part 1", r'^#\s+Building Social Media Content Automation Workflow \| Part 1\s*$'),
    ("Building Social Media Content Automation Workflow | Part 2", r'^#\s+Building AI-Powered Social Media Content Automation Workflow \| Part 2\s*$'),
    ("Advanced Prompt Engineering", r'^#\s+Advanced Prompt Engineering\s*$'),
    ("Build Your Own AI News Summarizer | Part 1", r'^#\s+Build Your Own AI News Summarizer \| Part 1\s*$'),
    ("Build Your Own AI News Summarizer | Part 2", r'^#\s+Build Your Own AI News Summarizer \| Part 2\s*$'),
    ("Build Your Own AI News Summarizer | Part 3", r'^#\s+Build Your Own AI News Summarizer \| Part 3\s*$'),
    ("Productivity Power-Up with AI Tools | Part 2", r'^#\s+Productivity Power-Up with AI Tools \| Part 2\s*$'),
    ("Mastering Image Generation", r'^#\s+Mastering Image Generation\s*$'),
    ("Setting Up Your Kaggle Environment", r'^#\s+Introduction\b[\s\S]{0,120}?In this unit, we will learn how to access free'),
    ("Mastering Image Generation with Stable Diffusion", r'^#\s+Mastering Image Generation with Stable Diffusion\s*$'),
    ("Introduction to AI Agents", r'^#\s+Introduction to AI Agents\s*$'),
    ("Mastering AI Audio Generation", r'^#\s+Mastering AI Audio Generation\s*$'),
    ("Building No-Code Applications with AI", r'^#\s+Building No-Code Applications with AI\s*$'),
    ("Mastering AI Audio Generation using F5-TTS", r'^#\s+Introduction\b[\s\S]{0,120}?In the previous unit we understand the core concepts'),
    ("Building a Learning Path Generator", r'^#\s+Building a Learning Path Generator\s*$'),
    ("Build Your Own AI Shopping Assistant | Part 1", r'^#\s+Build Your Own AI Shopping Assistant \| Part 1\s*$'),
    ("Build Your Own AI Shopping Assistant | Part 2", r'^#\s+Build Your Own AI Shopping Assistant \| Part 2\s*$'),
    ("Building an Agent with Memory", r'^#\s+Building an Agent with Memory\s*$'),
    ("Introduction to Model Context Protocol | Part 1", r'^#\s+Introduction to Model Context Protocol \| Part 1\s*$'),
    ("Introduction to Model Context Protocol | Part 2", r'^#\s+Introduction to Model Context Protocol \| Part 2\s*$'),
]

# llm_apps: canonical session name -> the exact `# <heading>` text that starts it.
LLM_APPS_ANCHORS = {
    "Introduction to Google Colab": "Introduction to Google Colab",
    "Introduction to Third Party Packages": "Introduction to Third-Party Packages in Python",
    "Introduction to Flask": "Introduction to Flask",
    "Building Rest APIs using Flask": "Building Rest APIs using Flask",
    "Integrating Flask APIs in Frontend": "Integrating Flask APIs in Frontend",
    "Building LLM Applications Using Python | Part 1": "Building LLM Applications using Python | Part 1",
    "Building LLM Applications Using Python | Part 2": "Building LLM Applications using Python | Part 2",
    "Building UI for LLM Applications": "Building UI for LLM Applications",
    "Deploying LLM Applications": "Deploying LLM Applications",
    "Understanding How LLMs Work | Part 1": "Understanding How LLMs Work | Part 1",
    "Understanding How LLMs Work | Part 2": "Understanding How LLMs Work | Part 2",
    "Tool use & Function Calling in LLMs": "Tool Use & Function Calling in LLMs",
    "Effective Prompting Techniques": "Effective Prompting Techniques",
    "Introduction to Langchain": "Introduction to LangChain",
    "Introduction to Retrieval-Augmented Generation | Part 1": "Introduction to Retrieval-Augmented Generation | Part 1",
    "Introduction to Retrieval-Augmented Generation | Part 2": "Introduction to Retrieval-Augmented Generation | Part 2",
    "Building AI Agents using Langchain": "Building AI Agents Using LangChain",
    "Building Memory Agents": "Building Memory Agent using Langchain",
    "AI in the Real World": "AI In The Real World",
    "Building an AI-Powered Conversational Interview Assistant | Part 1": "Building an AI-Powered Conversational Interview Assistant",
    "Building an AI-Powered Conversational Interview Assistant | Part 2": "Building an AI-Powered Conversational Interview Assistant | Part 2",
    "Introduction to Context Engineering": "Introduction to Context Engineering",
    "Integrating MCP": "Integrating MCP",
    "Building Multi Agent Systems Using Crew AI": "Building Multi Agent Systems Using Crew AI",
    "Building a Game Development Crew": "Building a Game Development Crew",
    "Introduction to LLM Evaluation | Part 1": "Introduction to LLM Application Evaluation | Part 1",
    "Introduction to LLM Evaluation | Part 2": "Introduction to LLM Application Evaluation | Part 2",
    "Running Models Locally": "Running Models Locally",
    "Fine-Tuning LLMs | Part 1": "Fine-Tuning LLMs",
}


def build_gen_ai(text: str) -> dict[str, str]:
    """Anchor on each session's `# <Title>` heading; slice between anchors in file order."""
    positions: list[tuple[int, str]] = []
    for session, pat in GEN_AI_ANCHORS:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            positions.append((m.start(), session))
        else:
            print(f"  WARNING: anchor not found for '{session}'")
    positions.sort()
    out: dict[str, str] = {}
    for i, (start, session) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        out[session] = text[start:end].strip()[:MAX_CHARS]
    return out


def build_llm_apps(text: str) -> dict[str, str]:
    """Anchor on exact `# <heading>` lines that match canonical session starts."""
    # Find the line position of each anchor heading (single `#`, optional quote).
    positions: list[tuple[int, str]] = []
    for session, heading in LLM_APPS_ANCHORS.items():
        pat = r'^"?#\s+' + re.escape(heading) + r'\s*$'
        m = re.search(pat, text, re.MULTILINE)
        if m:
            positions.append((m.start(), session))
        else:
            print(f"  WARNING: anchor not found for '{session}' ({heading!r})")
    positions.sort()
    out: dict[str, str] = {}
    for i, (start, session) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        out[session] = text[start:end].strip()[:MAX_CHARS]
    return out


def main():
    session_map: dict[str, str] = {}

    if GEN_AI_RM.exists():
        session_map.update(build_gen_ai(GEN_AI_RM.read_text(encoding="utf-8")))
    else:
        print(f"WARNING: {GEN_AI_RM} not found")

    if LLM_APPS_RM.exists():
        session_map.update(build_llm_apps(LLM_APPS_RM.read_text(encoding="utf-8")))
    else:
        print(f"WARNING: {LLM_APPS_RM} not found")

    OUT_PATH.write_text(json.dumps(session_map, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nWrote {len(session_map)} sessions → {OUT_PATH}")
    for name in session_map:
        print(f"  · {name}  ({len(session_map[name])} chars)")


if __name__ == "__main__":
    main()
