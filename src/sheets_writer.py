"""Google Sheets Writer — writes CuratedOutput to a Google Sheet using OAuth."""

import os
import json
import uuid
import gspread
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from src.models import CuratedOutput, QualityReport, strip_question_prefix
from src.config import PROJECT_ROOT
from src.data_loader import get_topic_for_session
from datetime import datetime


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TOKEN_PATH = PROJECT_ROOT / "token.json"

# Standard NxtMock unit overview card (LearningResources.Content), verbatim from the unit template.
# {INTERVIEW_ID} is substituted with this run's interview_id so the "Start Test" button deep-links correctly.
_OVERVIEW_CARD = """## **Overview**
* **Test Name:** NxtMock
* **Duration:** 30 minutes
* **Attempts:** 5 total (only 1 attempt active at a time)
* **Mode:** **Proctored** (webcam & screen monitored)

---

##  **System Setup**

Before starting, ensure:

1. **Stable Internet:** Minimum 2 Mbps connection.
2. **Device:** Laptop or desktop (avoid mobile/tablet).
3. **Browser:** Latest version of **Google Chrome**.
4. **Webcam & Mic:** Must be **ON** and functional throughout.
5. **Do not open new tabs, switch windows, or use incognito mode**—this triggers a violation warning.

---

<a href="https://nxtinterview-beta.earlywave.in/interview/{INTERVIEW_ID}" target="_blank">
  <button style="padding:10px 20px; font-size:16px; background-color:#2563eb; color:white; border:none; border-radius:6px; cursor:pointer;">
    Start Test
  </button>
</a>
"""


def _get_gspread_client() -> gspread.Client:
    """Authenticate with Google Sheets API using OAuth (client ID + secret from .env)."""
    creds = None

    # Check for cached token
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    # If no valid creds, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_id = os.getenv("GOOGLE_CLIENT_ID") or os.getenv("Client_ID")
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET") or os.getenv("Client_Secret")

            if not client_id or not client_secret:
                raise ValueError(
                    "Google OAuth credentials must be set in .env "
                    "(GOOGLE_CLIENT_ID / Client_ID and GOOGLE_CLIENT_SECRET / Client_Secret)"
                )

            client_config = {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }

            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=8090)

        # Save token for next run
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return gspread.authorize(creds)


def write_to_sheets(
    output: CuratedOutput,
    report: QualityReport,
    session_name: str,
    run_id: str,
    category: str = "GEN_AI",
) -> str:
    """Write CuratedOutput to a new Google Sheet. Returns the Sheet URL.

    `category` (from the course, e.g. GEN_AI / FULL_STACK) drives the topic,
    framework and Tags branding in the portal export.
    """
    client = _get_gspread_client()
    cat = (category or "GEN_AI").strip().upper().replace(" ", "_")

    # Per-run IDs and derived values — include coding-question IDs in the count
    org_id = str(uuid.uuid4())
    interview_id = str(uuid.uuid4())
    all_ids = ([q.question_id for q in output.question_details]
               + [q.id for q in output.coding_questions])
    question_ids_str = ", ".join(all_ids)
    q_count = len(all_ids)

    # Create new spreadsheet — name: "<Topic> - <session(s)> (NxtMock)"
    topic = get_topic_for_session(session_name)
    title = f"{topic} - {session_name} (NxtMock)" if topic else f"{session_name} (NxtMock)"
    spreadsheet = client.create(title)

    # --- Tab 1: QuestionDetails ---
    ws_qd = spreadsheet.sheet1
    ws_qd.update_title("QuestionDetails")

    qd_headers = [
        "question_id", "category", "content", "topic", "sub_topic",
        "difficulty", "language", "framework", "tool", "asked_in_company",
    ]
    qd_rows = [qd_headers]
    for q in output.question_details:
        qd_rows.append([
            q.question_id, q.category, strip_question_prefix(q.content), cat,
            q.sub_topic or "", (q.difficulty or "").upper(),
            q.language or "", cat, q.tool or "",
            q.attribution,
        ])

    if qd_rows:
        ws_qd.update(range_name="A1", values=qd_rows)

    # --- Tab 2: Organisation ---
    ws_org = spreadsheet.add_worksheet(title="Organisation", rows=2, cols=3)
    ws_org.update(range_name="A1", values=[
        ["org_id", "name", "logo_utl"],
        [org_id, "NxtWave", "https://nxtwave-website-media-files.s3.ap-south-1.amazonaws.com/ccbp-website/Nxtwave_Colored.svg"],
    ])

    # --- Tab 3: InterviewMinimalConfig ---
    ws_imc = spreadsheet.add_worksheet(title="InterviewMinimalConfig", rows=25, cols=13)
    imc_rows = [
        ["org_id", org_id],
        ["interview_id", interview_id],
        ["title", "MOCK INTERVIEW"],
        ["description", "This interview focuses on assessing candidates' knowledge applicability, hands-on practical experience, and overall knowledge testing."],
        ["max_attempts_allowed", 5],
        ["duration_in_secs", 1800],
        ["time_gap", 0],
        ["video_enabled", True],
        ["is_proctoring_enabled", True],
        ["category", "TESTING_CATEGORY"],
        ["Tags", f"NIAT_{cat}"],
        ["visibility", "should_show_report"],
        [None, True],
        ["slot_start_datetime", None],
        ["slot_end_datetime", None],
        ["lp_enroll_plans_supported", "NIAT"],
        # assess_entity table header
        ["assess_entity", "no_of_questions", "language", "topic", "sub_topic",
         "difficulty", "asked_in_company", "framework", "tool", "prompt_name_enum",
         "question_ids", "source_content", "preferred_language"],
        # SELF_INTRO row (keep as-is)
        ["SELF_INTRO", 1, None, "SELF_INTRO"],
        # GEN_AI row (replaces JAVA; question_ids populated from QuestionDetails)
        [cat, q_count, cat, None, None, None, None, None, None, None, question_ids_str],
    ]
    ws_imc.update(range_name="A1", values=imc_rows)

    # --- Unit-import scaffolding: ResourcesData / Units / LearningResourceSet / LearningResources ---
    # These wire the exported sheet up as a full importable LMS "unit" (the newer NxtMock unit format).
    # All entities are cross-linked by consistently-generated UUIDs, mirroring the unit template.
    unit_id = str(uuid.uuid4())            # unit_id == UNIT resource_id == learning-resource-set id
    common_unit_id = str(uuid.uuid4())
    child_resource_id = str(uuid.uuid4())  # child resource referenced in the dependency graph
    lr_uuid = str(uuid.uuid4())            # the LearningResources row referenced by the set

    ws_rd = spreadsheet.add_worksheet(title="ResourcesData", rows=4, cols=10)
    ws_rd.update(range_name="A1", values=[
        ["resource_id", "resource_type", "dependent_resource_count", "dependent_resources",
         "dependent_reason_display_text", "parent_resource_count", "child_order", "parent_resources",
         "auto_unlock", "is_primary"],
        [unit_id, "UNIT", 0, "", "", 1, "", "", True, ""],
        ["", "", "", "", "", "", 20, child_resource_id, "", True],
    ])

    ws_units = spreadsheet.add_worksheet(title="Units", rows=2, cols=5)
    ws_units.update(range_name="A1", values=[
        ["unit_id", "common_unit_id", "unit_type", "duration_in_sec", "unit_tags"],
        [unit_id, common_unit_id, "LEARNING_SET", "", "MOCK_TEST_EVALUATION"],
    ])

    ws_lrs = spreadsheet.add_worksheet(title="LearningResourceSet", rows=3, cols=5)
    ws_lrs.update(range_name="A1", values=[
        ["learning resource set id", "learning resource set name", "learning resources count",
         "learning resource ids", "order"],
        [unit_id, "NxtMock", 1, "", ""],
        ["", "", "", lr_uuid, 1],
    ])

    ws_lr = spreadsheet.add_worksheet(title="LearningResources", rows=2, cols=15)
    ws_lr.update(range_name="A1", values=[
        ["learning_resource_uuid", "Title", "Content", "content_format", "content_language",
         "multimedia_count", "multimedia_format", "total_duration", "multimedia_url", "thumbnail_url",
         "highlights_count", "\nduration in sec", "content", "\ntitle", "learning_resource_type"],
        [lr_uuid, "NxtMock", _OVERVIEW_CARD.replace("{INTERVIEW_ID}", interview_id), "MARKDOWN",
         "ENGLISH", 0, "", "", "", "", 0, "", "", "", "DEFAULT"],
    ])

    # --- Tab 4: CodeSnippet (only if snippets exist) ---
    if output.code_snippets:
        ws_cs = spreadsheet.add_worksheet(
            title="CodeSnippet",
            rows=len(output.code_snippets) + 1,
            cols=3,
        )
        cs_rows = [["code_id", "code_content", "Language"]]
        for s in output.code_snippets:
            cs_rows.append([s.code_id, s.code_content[:1000], s.language])
        ws_cs.update(range_name="A1", values=cs_rows)

    # --- Tab 5: CodingQuestion (only if coding questions exist) ---
    if output.coding_questions:
        ws_cq = spreadsheet.add_worksheet(
            title="CodingQuestion",
            rows=len(output.coding_questions) + 1,
            cols=15,
        )
        cq_headers = [
            "id", "category", "title", "content", "code_id", "topic", "sub_topic",
            "difficulty", "language", "framework", "tool", "asked_in_company",
        ]
        cq_rows = [cq_headers]
        for q in output.coding_questions:
            cq_rows.append([
                q.id, q.category, q.title, strip_question_prefix(q.content[:1000]), q.code_id or "",
                cat, q.sub_topic or "", (q.difficulty or "").upper(), q.language,
                cat, q.tool or "", q.attribution,
            ])
        ws_cq.update(range_name="A1", values=cq_rows)

    # --- Tab 6: CodeAnalysisQuestion (headers only — generation not yet wired) ---
    ws_caq = spreadsheet.add_worksheet(title="CodeAnalysisQuestion", rows=2, cols=6)
    ws_caq.update(range_name="A1", values=[
        ["question_id", "tag_name", "content", "code_id", "Title", "correct_answer"],
    ])

    return spreadsheet.url
