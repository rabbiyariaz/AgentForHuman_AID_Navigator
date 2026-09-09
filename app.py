"""AID Navigator — caseworker decision-support interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from agent.extraction_agent import run_extraction_pipeline
from main import DEMO_CASES
from triage.ranking import classify_case, rank_cases


st.set_page_config(
    page_title="AID Navigator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --ink:#102a27;
            --ink-soft:#27413d;
            --muted:#5b716d;
            --paper:#f3f7f4;
            --panel:#ffffff;
            --line:#d5e2dc;
            --teal:#087f74;
            --teal-dark:#06685f;
            --mint:#dff4ed;
            --mint-strong:#c8ebe0;
            --coral:#c9533e;
            --coral-bg:#fff0ec;
            --amber:#a66b09;
            --amber-bg:#fff6df;
            --blue:#356b91;
            --blue-bg:#edf5fa;
        }

        .stApp {
            background:var(--paper);
            color:var(--ink);
            font-family:'Space Grotesk', sans-serif;
        }

        [data-testid="stHeader"] { background:transparent; }
        #MainMenu, header, footer { visibility:hidden; }
        [data-testid="stToolbar"] { visibility:hidden; height:0; }
        [data-testid="stDecoration"] { display:none; }
        .block-container {
            max-width:1480px;
            padding:1.2rem 3.2rem 4rem;
        }

        /* ---------- Brand / header ---------- */
        .brand-header {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:2rem;
            padding:1rem 0 1.25rem;
            border-bottom:1px solid var(--line);
            margin-bottom:2rem;
        }
        .brand-main {
            display:flex;
            align-items:center;
            gap:.8rem;
        }
        .brand-star {
            color:var(--teal);
            font-size:2rem;
            line-height:1;
        }
        .brand-name {
            color:var(--ink);
            font-size:2rem;
            line-height:1;
            font-weight:700;
            letter-spacing:-.04em;
        }
        .brand-sub {
            margin-top:.38rem;
            color:var(--muted);
            font-family:'DM Mono',monospace;
            font-size:.68rem;
            letter-spacing:.08em;
            text-transform:uppercase;
        }
        .online {
            display:flex;
            align-items:center;
            gap:.55rem;
            color:var(--ink-soft);
            font-size:.85rem;
            white-space:nowrap;
            background:#fff;
            border:1px solid var(--line);
            border-radius:999px;
            padding:.55rem .8rem;
        }
        .online-dot {
            width:9px;
            height:9px;
            border-radius:50%;
            background:#2eae7d;
            box-shadow:0 0 0 4px #d9f3e8;
        }

        /* ---------- Tabs ---------- */
        button[data-baseweb="tab"] {
            font-family:'Space Grotesk',sans-serif !important;
            font-size:1rem !important;
            font-weight:600 !important;
            color:var(--muted) !important;
            padding:0.75rem 1.2rem !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color:var(--teal) !important;
        }

        /* ---------- Hero ---------- */
        .eyebrow {
            color:var(--teal);
            font-family:'DM Mono',monospace;
            font-size:.74rem;
            letter-spacing:.12em;
            text-transform:uppercase;
            font-weight:500;
        }
        .hero {
            padding:1.2rem 0 .8rem;
            max-width:930px;
        }
        .hero h1 {
            color:var(--ink);
            font-size:clamp(2.8rem,5vw,4.6rem);
            line-height:.98;
            letter-spacing:-.055em;
            margin:.6rem 0 .9rem;
        }
        .hero p {
            color:var(--ink-soft);
            font-size:1.12rem;
            line-height:1.6;
            max-width:820px;
        }

        /* ---------- Workflow ---------- */
        .workflow {
            display:grid;
            grid-template-columns:repeat(5,1fr);
            gap:.6rem;
            margin:1.2rem 0 1.5rem;
        }
        .workflow-step {
            border:1px solid var(--line);
            background:#fff;
            border-radius:8px;
            padding:.7rem .75rem;
            min-height:64px;
        }
        .workflow-step.active {
            border-color:#67b7a8;
            background:var(--mint);
            box-shadow:0 3px 10px rgba(8,127,116,.08);
        }
        .workflow-step.complete {
            border-color:#b8dcd3;
            background:#f8fcfa;
        }
        .workflow-num {
            display:block;
            color:var(--teal);
            font-family:'DM Mono',monospace;
            font-size:.68rem;
            margin-bottom:.25rem;
        }
        .workflow-label {
            color:var(--ink);
            font-size:.86rem;
            font-weight:600;
        }
        .workflow-help {
            color:var(--muted);
            font-size:.78rem;
            margin-top:.2rem;
        }

        /* ---------- Panels ---------- */
        .surface {
            background:var(--panel);
            border:1px solid var(--line);
            border-radius:12px;
            padding:1.35rem 1.4rem;
            box-shadow:0 8px 24px rgba(16,42,39,.045);
        }
        .surface-title {
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap:1rem;
            margin-bottom:1.1rem;
        }
        .surface-title h3 {
            margin:0;
            color:var(--ink);
            font-size:1.25rem;
            letter-spacing:-.02em;
        }
        .surface-title p {
            margin:.25rem 0 0;
            color:var(--muted);
            font-size:.84rem;
        }
        .small-tag {
            color:var(--teal);
            background:var(--mint);
            border:1px solid var(--mint-strong);
            border-radius:999px;
            padding:.35rem .6rem;
            font-family:'DM Mono',monospace;
            font-size:.70rem;
            white-space:nowrap;
        }

        /* Make Streamlit form labels and input text readable */
        label, [data-testid="stWidgetLabel"] p {
            color:var(--ink-soft) !important;
            font-size:.9rem !important;
            font-weight:600 !important;
        }
        input, textarea {
            font-size:.95rem !important;
        }
        textarea {
            min-height:145px !important;
        }

        /* ---------- Upload area ---------- */
        [data-testid="stFileUploader"] {
            border:1.5px dashed #8ebbb0;
            border-radius:9px;
            padding:.25rem;
            background:#f9fcfa;
        }
        [data-testid="stFileUploader"] section {
            padding:.55rem .7rem !important;
        }
        [data-testid="stFileUploader"] button {
            font-size:.85rem !important;
        }
        .upload-note {
            color:var(--muted);
            font-size:.78rem;
            line-height:1.45;
            margin-top:-.35rem;
            margin-bottom:.8rem;
        }

        /* ---------- Decision waiting state ---------- */
        .waiting {
            border:1px solid var(--line);
            background:#f7faf8;
            border-radius:9px;
            padding:1.15rem;
        }
        .waiting-title {
            color:var(--ink);
            font-size:1.05rem;
            font-weight:700;
            margin-bottom:.3rem;
        }
        .waiting-copy {
            color:var(--muted);
            font-size:.86rem;
            line-height:1.5;
            margin-bottom:.9rem;
        }
        .waiting-grid {
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:.55rem;
        }
        .waiting-item {
            background:#fff;
            border:1px solid var(--line);
            border-radius:7px;
            padding:.7rem;
        }
        .waiting-item strong {
            display:block;
            color:var(--teal);
            font-size:.76rem;
            margin-bottom:.2rem;
        }
        .waiting-item span {
            color:var(--muted);
            font-size:.72rem;
            line-height:1.35;
        }

        /* ---------- Decision output ---------- */
        .decision-header {
            display:flex;
            align-items:flex-start;
            justify-content:space-between;
            gap:1rem;
            margin-bottom:.8rem;
        }
        .decision-kicker {
            color:var(--muted);
            font-family:'DM Mono',monospace;
            font-size:.66rem;
            letter-spacing:.09em;
            text-transform:uppercase;
        }
        .decision-title {
            color:var(--ink);
            font-size:1.45rem;
            font-weight:700;
            margin-top:.2rem;
        }
        .badge-row {
            display:flex;
            gap:.4rem;
            flex-wrap:wrap;
            justify-content:flex-end;
        }
        .badge {
            border-radius:999px;
            padding:.35rem .6rem;
            font-family:'DM Mono',monospace;
            font-size:.64rem;
            white-space:nowrap;
        }
        .badge.confidence {
            color:var(--teal);
            background:var(--mint);
            border:1px solid #a9d9cb;
        }
        .badge.urgent {
            color:var(--coral);
            background:var(--coral-bg);
            border:1px solid #efb9ab;
        }
        .status {
            border-radius:8px;
            padding:1rem 1.1rem;
            margin-bottom:.9rem;
        }
        .status.ready {
            border:1px solid #b8dfd5;
            background:var(--mint);
        }
        .status.review {
            border:1px solid #efb9ab;
            background:var(--coral-bg);
        }
        .status.followup {
            border:1px solid #ead49c;
            background:var(--amber-bg);
        }
        .status h3 {
            color:var(--ink);
            margin:0 0 .2rem;
            font-size:1.02rem;
        }
        .status p {
            color:var(--ink-soft);
            margin:0;
            font-size:.84rem;
            line-height:1.45;
        }
        .decision-rule {
            border-top:1px solid var(--line);
            margin-top:1rem;
            padding-top:.9rem;
        }
        .rule-label {
            display:block;
            color:var(--teal);
            font-family:'DM Mono',monospace;
            font-size:.72rem;
            letter-spacing:.08em;
            text-transform:uppercase;
            margin-bottom:.3rem;
        }
        .rule-text {
            color:var(--ink-soft);
            font-size:.82rem;
            line-height:1.45;
        }

        /* ---------- Metrics ---------- */
        .metric {
            background:#f2f7f4;
            border:1px solid #e0ebe6;
            border-radius:8px;
            padding:.8rem .9rem;
            min-height:76px;
        }
        .metric-label {
            color:var(--muted);
            font-family:'DM Mono',monospace;
            font-size:.72rem;
            letter-spacing:.06em;
            text-transform:uppercase;
        }
        .metric-value {
            color:var(--ink);
            font-size:1.28rem;
            font-weight:700;
            margin-top:.28rem;
        }

        /* ---------- Evidence ---------- */
        .section-heading {
            color:var(--ink);
            font-size:1.2rem;
            font-weight:700;
            margin:1.5rem 0 .7rem;
        }
        .evidence-row {
            display:grid;
            grid-template-columns:1.05fr 1fr .8fr;
            gap:1rem;
            align-items:center;
            padding:.72rem 0;
            border-bottom:1px solid #edf1ee;
        }
        .evidence-field { color:var(--ink); font-weight:600; font-size:.96rem; }
        .evidence-value { color:var(--ink-soft); font-size:.96rem; }
        .evidence-meta {
            color:var(--muted);
            font-family:'DM Mono',monospace;
            font-size:.78rem;
            text-align:right;
        }

        /* ---------- Queue ---------- */
        .queue-hero {
            display:grid;
            grid-template-columns:1.35fr .65fr;
            gap:1.2rem;
            align-items:stretch;
            margin:1rem 0 1.2rem;
        }
        .queue-explainer {
            background:#fff;
            border:1px solid var(--line);
            border-radius:10px;
            padding:1.15rem 1.25rem;
        }
        .queue-explainer h2 {
            color:var(--ink);
            font-size:1.65rem;
            margin:0 0 .45rem;
            letter-spacing:-.03em;
        }
        .queue-explainer p {
            color:var(--ink-soft);
            font-size:1.02rem;
            line-height:1.6;
            margin:0;
        }
        .queue-purpose {
            background:var(--ink);
            color:#fff;
            border-radius:10px;
            padding:1.15rem 1.2rem;
        }
        .queue-purpose .purpose-label {
            color:#9de0d2;
            font-family:'DM Mono',monospace;
            font-size:.72rem;
            letter-spacing:.08em;
        }
        .queue-purpose h3 {
            color:#fff;
            margin:.35rem 0 .35rem;
            font-size:1.05rem;
        }
        .queue-purpose p {
            color:#d9e8e4;
            font-size:.88rem;
            line-height:1.5;
            margin:0;
        }
        .queue-steps {
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:.65rem;
            margin-bottom:1.2rem;
        }
        .queue-step {
            background:#fff;
            border:1px solid var(--line);
            border-radius:8px;
            padding:.75rem;
        }
        .queue-step strong {
            display:block;
            color:var(--teal);
            font-size:.8rem;
            margin-bottom:.2rem;
        }
        .queue-step span {
            color:var(--muted);
            font-size:.84rem;
            line-height:1.45;
        }
        .legend {
            display:flex;
            gap:1rem;
            flex-wrap:wrap;
            margin:.5rem 0 1rem;
        }
        .legend-item {
            display:flex;
            align-items:center;
            gap:.4rem;
            color:var(--ink-soft);
            font-size:.78rem;
        }
        .legend-dot {
            width:9px;
            height:9px;
            border-radius:50%;
            background:var(--teal);
        }
        .legend-dot.amber { background:var(--amber); }
        .legend-dot.coral { background:var(--coral); }

        .queue-card {
            border:1px solid var(--line);
            border-left:5px solid var(--teal);
            background:#fff;
            border-radius:9px;
            padding:.78rem .95rem;
            margin-bottom:.58rem;
            box-shadow:0 4px 12px rgba(16,42,39,.035);
        }
        .queue-card.review { border-left-color:var(--coral); }
        .queue-card.followup { border-left-color:var(--amber); }
        .queue-card.ineligible { border-left-color:#6f7d79; }
        .queue-top {
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap:1rem;
        }
        .queue-name {
            color:var(--ink);
            font-size:1rem;
            font-weight:700;
        }
        .queue-tier {
            color:var(--muted);
            font-family:'DM Mono',monospace;
            font-size:.72rem;
            white-space:nowrap;
        }
        .queue-reason {
            color:var(--ink-soft);
            margin:.3rem 0 0;
            font-size:.9rem;
            line-height:1.45;
        }
        .queue-detail {
            display:flex;
            gap:.45rem;
            flex-wrap:wrap;
            margin-top:.55rem;
        }
        .queue-chip {
            color:var(--muted);
            background:#f7faf8;
            border:1px solid var(--line);
            border-radius:999px;
            padding:.3rem .55rem;
            font-family:'DM Mono',monospace;
            font-size:.62rem;
        }
        .queue-chip.action {
            color:var(--teal);
            background:var(--mint);
            border-color:#a9d9cb;
            font-weight:500;
        }

        /* ---------- Buttons ---------- */
        div[data-testid="stButton"] button[kind="primary"] {
            background:var(--teal);
            border-color:var(--teal);
            color:#fff;
            border-radius:7px;
            font-weight:700;
            min-height:46px;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background:var(--teal-dark);
            border-color:var(--teal-dark);
        }

        .demo-note {
            color:var(--muted);
            font-size:.84rem;
            margin-top:.4rem;
        }

        @media (max-width: 900px) {
            .block-container { padding:1rem 1rem 3rem; }
            .brand-header { display:block; }
            .online { margin-top:1rem; width:max-content; }
            .workflow { grid-template-columns:1fr 1fr; }
            .queue-hero, .queue-steps { grid-template-columns:1fr; }
            .waiting-grid { grid-template-columns:1fr; }
        }

        @media (max-width: 600px) {
            .brand-name { font-size:1.6rem; }
            .hero h1 { font-size:2.7rem; }
            .workflow { grid-template-columns:1fr; }
            .evidence-row { grid-template-columns:1fr; gap:.15rem; }
            .evidence-meta { text-align:left; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_case(intake: str, uploads) -> dict:
    document_path = None

    if isinstance(uploads, (str, Path)):
        document_path = str(uploads)
    elif uploads:
        if not isinstance(uploads, list):
            uploads = [uploads]

        # The current extraction pipeline is fed text. Combine multiple
        # text-based supporting records into one clearly separated file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as handle:
            for upload in uploads:
                handle.write(
                    f"\n--- Supporting document: {upload.name} ---\n".encode("utf-8")
                )
                handle.write(upload.getvalue())
                handle.write(b"\n")
            document_path = handle.name

    return run_extraction_pipeline(
        intake,
        document_path_or_text=document_path,
    )


def status_copy(case: dict) -> tuple[str, str, str]:
    status = case.get("status")

    if status == "needs_review":
        return (
            "CASEWORKER REVIEW",
            "Conflicting evidence was found. A person should verify the source before a decision is made.",
            "review",
        )

    if status == "needs_followup":
        return (
            "NEEDS INFORMATION",
            "The case is not ready for a safe decision. A targeted follow-up question is available.",
            "followup",
        )

    triage = case.get("triage") or classify_case(case)
    crisis = (triage.get("eligibility") or {}).get("crisis", {})

    if crisis.get("eligible"):
        return (
            "CRISIS ELIGIBLE",
            "The household meets the emergency assistance pathway.",
            "ready",
        )

    regular = (triage.get("eligibility") or {}).get("regular", {})

    if regular.get("eligible"):
        return (
            "REGULAR ELIGIBLE",
            "The household meets the standard income eligibility rule.",
            "ready",
        )

    return (
        "NOT ELIGIBLE",
        "The available evidence does not qualify the case under the deterministic rule.",
        "review",
    )


def friendly_reason(case: dict) -> str:
    """Turn internal rule names into something a caseworker can understand."""
    status = case.get("status")
    triage = case.get("triage") or classify_case(case)

    if status == "needs_review":
        contradictions = case.get("contradictions", [])
        if contradictions:
            fields = ", ".join(
                item.get("field", "a field").replace("_", " ")
                for item in contradictions
            )
            return f"Two sources disagree about {fields}. Human verification is required."
        return "The evidence contains a conflict that should be reviewed by a caseworker."

    if status == "needs_followup":
        missing = triage.get("missing_fields") or case.get("missing_fields") or []
        if missing:
            readable = ", ".join(str(x).replace("_", " ") for x in missing)
            return f"Important information is missing: {readable}. Ask the resident before deciding."
        return "Important information is missing. Ask the next targeted question before deciding."

    eligibility = triage.get("eligibility") or {}
    crisis = eligibility.get("crisis", {})
    regular = eligibility.get("regular", {})

    if crisis.get("eligible"):
        arrears = crisis.get("arrearage_after_benefit")
        if arrears is not None:
            return (
                f"Crisis pathway qualifies the household; remaining arrears are "
                f"${arrears:,.0f} and the emergency condition is present."
            )
        return "The household qualifies for the crisis assistance pathway."

    if regular.get("eligible"):
        limit = regular.get("income_limit")
        income = regular.get("annual_income")
        if income is not None and limit is not None:
            return (
                f"Annual income of ${income:,.0f} is within the "
                f"${limit:,.0f} household income limit."
            )
        return "The household meets the standard income eligibility rule."

    income = regular.get("annual_income")
    limit = regular.get("income_limit")
    if income is not None and limit is not None:
        return (
            f"Annual income of ${income:,.0f} is above the "
            f"${limit:,.0f} household income limit."
        )

    return "The deterministic eligibility rule did not qualify this case."


def render_workflow(active_step: int) -> None:
    workflow = [
        ("01", "INTAKE", "Resident story"),
        ("02", "EXTRACT", "Find key facts"),
        ("03", "VERIFY", "Check evidence"),
        ("04", "DETERMINE", "Apply rules"),
        ("05", "PRIORITIZE", "Set next action"),
    ]

    items = []
    for number, (index, label, help_text) in enumerate(workflow, 1):
        state = (
            "active"
            if number == active_step
            else "complete"
            if number < active_step
            else ""
        )
        items.append(
            f'<div class="workflow-step {state}">'
            f'<span class="workflow-num">{index}</span>'
            f'<div class="workflow-label">{label}</div>'
            f'<div class="workflow-help">{help_text}</div>'
            f"</div>"
        )

    st.markdown(
        f'<div class="workflow">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


def render_evidence(case: dict) -> None:
    fields = (
        case.get("extraction_output", {}).get("extracted", {})
        or case.get("partial_fields", {})
    )
    audit = (
        case.get("extraction_output", {}).get("audit", {})
        or case.get("partial_audit", {})
    )

    labels = {
        "household_size": "Household size",
        "annual_income": "Annual income",
        "home_type": "Home type",
        "heating_fuel_type": "Heating fuel",
        "disconnection_status": "Disconnected",
        "arrearage_amount": "Arrears",
        "fuel_tank_percent": "Tank level",
        "heat_included_in_rent": "Heat in rent",
        "applicant_name": "Applicant",
    }

    rows = []

    for key, value in fields.items():
        if value is None or key == "stated_deadline_or_urgency":
            continue

        display = (
            f"${value:,.0f}"
            if key in {"annual_income", "arrearage_amount"}
            else str(value)
        )

        if isinstance(value, bool):
            display = "Yes" if value else "No"

        meta = audit.get(key, {})
        source = meta.get("source", "text")
        confidence = f"{float(meta.get('confidence', 0)):.0%} confidence"

        rows.append(
            '<div class="evidence-row">'
            f'<div class="evidence-field">{labels.get(key, key.replace("_", " ").title())}</div>'
            f'<div class="evidence-value">{display}</div>'
            f'<div class="evidence-meta">{source} · {confidence}</div>'
            "</div>"
        )

    st.markdown(
        "".join(rows)
        or '<p style="color:var(--muted);font-size:.85rem">No evidence extracted yet.</p>',
        unsafe_allow_html=True,
    )


def render_decision(case: dict) -> None:
    triage = case.get("triage") or classify_case(case)
    title, subtitle, kind = status_copy(case)

    urgency = case.get("next_action", {}).get("urgency_flag", False)
    confidence = triage.get("confidence")

    badges = []

    if urgency:
        badges.append('<span class="badge urgent">URGENT</span>')

    if confidence is not None:
        badges.append(
            f'<span class="badge confidence">{float(confidence):.0%} confidence</span>'
        )

    st.markdown(
        '<div class="decision-header">'
        "<div>"
        '<div class="decision-kicker">CASE OUTCOME</div>'
        f'<div class="decision-title">{title}</div>'
        "</div>"
        f'<div class="badge-row">{"".join(badges)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="status {kind}">'
        f"<h3>{subtitle}</h3>"
        "<p>The evidence and next action are shown below so the caseworker can act quickly and audit the decision.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    agent_trace = case.get("agent_trace", [])
    if agent_trace:
        with st.expander("Agent activity", expanded=False):
            for event in agent_trace:
                st.write(f"• {event}")

    if case.get("status") == "needs_followup":
        st.markdown("**Next question(s) to ask**")
        questions = case.get("next_action", {}).get("followup_questions", [])
        for question in questions:
            st.info(question, icon="ℹ️")

    elif case.get("status") == "needs_review":
        st.markdown("**Why human review is required**")
        for item in case.get("contradictions", []):
            field = item.get("field", "Field").replace("_", " ").title()
            text_value = item.get("text_value")
            document_value = item.get("document_value")
            st.warning(
                f"**{field}:** application says `{text_value}`, "
                f"supporting document says `{document_value}`."
            )

    else:
        regular = triage["eligibility"]["regular"]
        crisis = triage["eligibility"]["crisis"]

        a, b, c = st.columns(3)
        a.markdown(
            f'<div class="metric"><div class="metric-label">REGULAR BENEFIT</div>'
            f'<div class="metric-value">${regular["benefit_amount"]:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
        b.markdown(
            f'<div class="metric"><div class="metric-label">INCOME LIMIT</div>'
            f'<div class="metric-value">${regular["income_limit"]:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
        c.markdown(
            f'<div class="metric"><div class="metric-label">CRISIS PATH</div>'
            f'<div class="metric-value">{"Yes" if crisis["eligible"] else "No"}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="decision-rule">'
            '<span class="rule-label">WHY THIS DECISION</span>'
            f'<span class="rule-text">{friendly_reason(case)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_intake() -> None:
    st.markdown(
        '<div class="eyebrow">INTAKE WORKSPACE · SINGLE CASE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero">'
        "<h1>Turn a hard story into the next right action.</h1>"
        "<p>Paste what the resident told you. AID Navigator extracts the important facts, checks supporting evidence, applies the protected eligibility rules, and tells you whether to decide, ask, or review.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    current_case = st.session_state.get("current_case")

    if not current_case:
        active_step = 1
    elif current_case.get("status") in {"needs_followup", "needs_review"}:
        active_step = 3
    else:
        active_step = 5

    render_workflow(active_step)

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.markdown(
            '<div class="surface">'
            '<div class="surface-title">'
            "<div><h3>Resident intake</h3>"
            "<p>Start with the resident's own words. Add supporting records when available.</p></div>"
            '<span class="small-tag">STEP 01 · INTAKE</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        name = st.text_input(
            "Applicant name",
            placeholder="e.g. Maria Lopez",
        )

        intake = st.text_area(
            "What is happening?",
            height=155,
            placeholder=(
                "Example: My power is scheduled to be shut off Friday. "
                "There are four people in my household and our annual income is $35,000..."
            ),
        )

        st.markdown("**Supporting evidence**")
        st.markdown(
            '<div class="upload-note">'
            "Attach one or more text-based supporting records. For example: income verification, "
            "utility bill details, or a shutoff notice. Multiple files are combined and kept clearly separated."
            "</div>",
            unsafe_allow_html=True,
        )

        uploads = st.file_uploader(
            "Add supporting document(s)",
            type=["txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="You can attach multiple .txt records.",
        )

        if uploads:
            st.caption(
                f"{len(uploads)} supporting document(s) attached: "
                + ", ".join(upload.name for upload in uploads)
            )

        analyze = st.button(
            "Analyze application  →",
            type="primary",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if analyze:
            if not intake.strip():
                st.error("Add an intake message before analyzing.")
            else:
                with st.spinner("Extracting evidence, verifying sources, and applying protected rules..."):
                    try:
                        result = run_case(intake, uploads)
                        result["display_name"] = (
                            name
                            or result.get("partial_fields", {}).get(
                                "applicant_name",
                                "Applicant",
                            )
                        )
                        st.session_state["current_case"] = result
                        st.rerun()
                    except Exception as exc:
                        st.error(f"The case could not be analyzed: {exc}")

    with right:
        st.markdown(
            '<div class="surface">'
            '<div class="surface-title">'
            "<div><h3>Decision workspace</h3>"
            "<p>This is where the system explains what should happen next.</p></div>"
            '<span class="small-tag">AUDITABLE OUTPUT</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        case = st.session_state.get("current_case")

        if case:
            render_decision(case)
        else:
            st.markdown(
                '<div class="waiting">'
                '<div class="waiting-title">Your case decision will appear here</div>'
                '<div class="waiting-copy">AID Navigator does not hide the important steps. After analysis, this panel will show the outcome, what evidence supports it, and the action a caseworker should take.</div>'
                '<div class="waiting-grid">'
                '<div class="waiting-item"><strong>EXTRACT</strong><span>Key facts found in the resident story and documents.</span></div>'
                '<div class="waiting-item"><strong>VERIFY</strong><span>Conflicts are stopped instead of being guessed.</span></div>'
                '<div class="waiting-item"><strong>ACT</strong><span>Decide, ask a question, or send the case to review.</span></div>'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("current_case"):
        st.markdown('<div class="section-heading">Evidence trail</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="surface">',
            unsafe_allow_html=True,
        )
        st.caption(
            "Use this trail to verify the facts behind the decision. Source and confidence are shown for every extracted field."
        )
        render_evidence(st.session_state["current_case"])
        st.markdown("</div>", unsafe_allow_html=True)


def render_queue() -> None:
    st.markdown(
        '<div class="eyebrow">CASEWORKER CONSOLE · TRIAGE QUEUE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="queue-hero">'
        '<div class="queue-explainer">'
        '<h2>Put the right case in front of the right person.</h2>'
        "<p>The triage queue is not another list of applications. It sorts cases by what needs attention first: urgent situations, missing information, evidence conflicts, and cases that are ready for a safe decision.</p>"
        "</div>"
        '<div class="queue-purpose">'
        '<div class="purpose-label">WHY TRIAGE?</div>'
        "<h3>Reduce the cases a human has to think through from scratch.</h3>"
        "<p>Urgent cases rise to the top. Missing information gets a clear question. Contradictory evidence is held for human review. Complete cases can move forward.</p>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="queue-steps">'
        '<div class="queue-step"><strong>01 · ACT</strong><span>Urgent or decision-ready cases are surfaced first.</span></div>'
        '<div class="queue-step"><strong>02 · ASK</strong><span>Incomplete cases tell the worker exactly what information is missing.</span></div>'
        '<div class="queue-step"><strong>03 · REVIEW</strong><span>Conflicting sources are paused instead of silently resolved.</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="legend">'
        '<span class="legend-item"><span class="legend-dot"></span>urgent / ready</span>'
        '<span class="legend-item"><span class="legend-dot amber"></span>missing information</span>'
        '<span class="legend-item"><span class="legend-dot coral"></span>human review</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="color:var(--muted);font-size:.92rem;margin:-.35rem 0 1rem;">'
        "<strong style='color:var(--ink-soft)'>How to read the queue:</strong> "
        "work from top to bottom. Each card answers three questions — "
        "<em>what happened, why is it here, and what should I do next?</em>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Load five example cases  →", type="primary"):
        with st.spinner("Processing the community queue..."):
            results = [
                run_case(item["text"], item["document"])
                for item in DEMO_CASES
            ]
            st.session_state["queue"] = rank_cases(results)
            st.rerun()

    st.markdown(
        '<div class="demo-note">Demo data only · the queue is ranked by urgency, evidence completeness, contradiction risk, and eligibility.</div>',
        unsafe_allow_html=True,
    )

    queue = st.session_state.get("queue")

    if not queue:
        st.markdown(
            '<div class="surface" style="margin-top:1rem">'
            '<strong style="color:var(--ink);font-size:1rem">No cases in the queue yet.</strong>'
            '<p style="color:var(--muted);font-size:.84rem;margin:.3rem 0 0">'
            "Run the five-case demo to see how AID Navigator turns raw applications into a prioritized worklist."
            "</p></div>",
            unsafe_allow_html=True,
        )
        return

    urgent = sum(
        bool(case.get("next_action", {}).get("urgency_flag"))
        and case.get("status") != "needs_review"
        for case in queue
    )
    review = sum(case["status"] == "needs_review" for case in queue)
    followup = sum(case["status"] == "needs_followup" for case in queue)

    a, b, c, d = st.columns(4)
    a.markdown(
        f'<div class="metric"><div class="metric-label">CASES PROCESSED</div><div class="metric-value">{len(queue)}</div></div>',
        unsafe_allow_html=True,
    )
    b.markdown(
        f'<div class="metric"><div class="metric-label">URGENT SIGNALS</div><div class="metric-value">{urgent}</div></div>',
        unsafe_allow_html=True,
    )
    c.markdown(
        f'<div class="metric"><div class="metric-label">NEED INFORMATION</div><div class="metric-value">{followup}</div></div>',
        unsafe_allow_html=True,
    )
    d.markdown(
        f'<div class="metric"><div class="metric-label">HUMAN REVIEWS</div><div class="metric-value">{review}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading" style="font-size:1.35rem;">Priority order</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Top = the case that deserves attention first. The reason below each case explains why."
    )

    for index, case in enumerate(queue, 1):
        title, subtitle, kind = status_copy(case)
        name = (
            case.get("extraction_output", {})
            .get("extracted", {})
            .get("applicant_name")
            or case.get("partial_fields", {}).get("applicant_name")
            or f"Case {index}"
        )

        triage = case["triage"]
        next_action = case.get("next_action", {})

        if case["status"] == "needs_followup":
            action = f"Ask {len(next_action.get('followup_questions', []))} question(s)"
        elif case["status"] == "needs_review":
            action = "Human verification"
        elif triage.get("eligibility", {}).get("crisis", {}).get("eligible"):
            action = "Crisis action"
        elif triage.get("eligibility", {}).get("regular", {}).get("eligible"):
            action = "Decision ready"
        else:
            action = "No benefit"

        urgency = "urgent" if next_action.get("urgency_flag") else "standard"
        card_kind = "ineligible" if title == "NOT ELIGIBLE" else kind
        chip_text = "review required" if case["status"] == "needs_review" else ("urgent signal" if urgency == "urgent" else "standard")

        st.markdown(
            f'<div class="queue-card {card_kind}">'
            '<div class="queue-top">'
            f'<span class="queue-name">{index:02d} · {name}</span>'
            f'<span class="queue-tier">RANK {index} · {title}</span>'
            "</div>"
            f'<p class="queue-reason">{friendly_reason(case)}</p>'
            '<div class="queue-detail">'
            f'<span class="queue-chip action">{action}</span>'
            f'<span class="queue-chip">{float(triage["confidence"]):.0%} confidence</span>'
            f'<span class="queue-chip">{chip_text}</span>'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()

    st.markdown(
        '<div class="brand-header">'
        '<div class="brand-main">'
        '<span class="brand-star">✦</span>'
        '<div><div class="brand-name">AID Navigator</div>'
        '<div class="brand-sub">Community Energy Assistance · DC LIHEAP FY26</div></div>'
        "</div>"
        '<div class="online"><span class="online-dot"></span>Agent online · rules protected</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    intake_tab, queue_tab = st.tabs(
        ["Intake workspace", "Triage queue"]
    )

    with intake_tab:
        render_intake()

    with queue_tab:
        render_queue()


if __name__ == "__main__":
    main()
