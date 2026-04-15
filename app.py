from __future__ import annotations

from dataclasses import dataclass
import re
from html import escape
from pathlib import Path

import streamlit as st

from fake_job_demo import (
    COUNT_LR_MODEL_ID,
    DEMO_MODEL_SPECS,
    FIELD_LABELS,
    MODEL_FAMILY_LEXICAL,
    MODEL_FAMILY_TRANSFORMER,
    MODEL_TEXT_FIELDS,
    MULTILINGUAL_PRIMARY_MODEL_ID,
    PredictionResult,
    TFIDF_LR_MODEL_ID,
    TermContribution,
    get_demo_model_spec,
    load_demo_model_state,
    run_demo_models_job_post,
    run_demo_models_raw_text,
)

st.set_page_config(
    page_title="Fake Job Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

INPUT_MODE_COMBINED = "combined_text"
INPUT_MODE_STRUCTURED = "structured_fields"

NON_LATIN_SCRIPT_RE = re.compile(
    r"[\u0400-\u052F\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]"
)

SAMPLE_POSTS = {
    "legit": {
        "label": "Legit Sample",
        "job_post": {
            "title": "Senior Data Analyst",
            "company_profile": (
                "Established healthcare analytics company supporting hospitals and payer "
                "operations across North America."
            ),
            "description": (
                "Build recurring finance and operations dashboards, investigate data quality "
                "issues, and partner with product managers to improve KPI definitions."
            ),
            "requirements": (
                "3+ years of SQL and Python analysis experience, strong communication skills, "
                "and experience presenting findings to non-technical stakeholders."
            ),
            "benefits": (
                "Medical, dental, retirement plan, education stipend, and hybrid work policy."
            ),
        },
    },
    "suspicious": {
        "label": "Suspicious Sample",
        "job_post": {
            "title": "Data Entry Receptionist",
            "company_profile": "Online financing office expanding globally.",
            "description": (
                "Earn money from home with fast onboarding and same-day payment. "
                "No experience needed. Contact our recruiting agent today to start immediately."
            ),
            "requirements": (
                "Must be motivated, available on Telegram, and willing to submit daily reports."
            ),
            "benefits": "Weekly cash wages, instant bonuses, and referral commissions.",
        },
    },
}

UI_MODEL_SPECS = tuple(
    spec
    for spec in DEMO_MODEL_SPECS
    if spec.model_id not in {TFIDF_LR_MODEL_ID, MULTILINGUAL_PRIMARY_MODEL_ID}
)
UI_MODEL_IDS = frozenset(spec.model_id for spec in UI_MODEL_SPECS)
UI_DEFAULT_MODEL_ID = COUNT_LR_MODEL_ID


@dataclass(frozen=True)
class ResultDisplayView:
    headline_label: str
    supporting_text: str
    confidence_label: str
    status_tone: str


def main() -> None:
    inject_styles()
    initialize_state()
    model_states = load_runtime_state()

    render_hero(model_states)

    left_col, right_col = st.columns([1.02, 0.98], gap="large")
    with left_col:
        should_predict = render_input_panel(model_states)
    with right_col:
        render_result_panel(model_states, should_predict)

    render_model_notes(model_states)


def initialize_state() -> None:
    st.session_state.setdefault("input_mode", INPUT_MODE_COMBINED)
    st.session_state.setdefault("selected_model_id", UI_DEFAULT_MODEL_ID)
    if st.session_state["selected_model_id"] not in UI_MODEL_IDS:
        st.session_state["selected_model_id"] = UI_DEFAULT_MODEL_ID
    st.session_state.setdefault("combined_text", "")
    st.session_state.setdefault("last_results", None)
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_input_mode", INPUT_MODE_COMBINED)
    st.session_state.setdefault("last_contains_non_latin", False)
    for field in MODEL_TEXT_FIELDS:
        st.session_state.setdefault(field, "")


def reset_result_state() -> None:
    st.session_state["last_results"] = None
    st.session_state["last_error"] = ""
    st.session_state["last_contains_non_latin"] = False


def load_runtime_state():
    return tuple(load_demo_model_state(spec.model_id) for spec in UI_MODEL_SPECS)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: rgba(255, 255, 255, 0.68);
            --surface-strong: rgba(255, 255, 255, 0.82);
            --ink: #142032;
            --ink-soft: #556174;
            --line: rgba(20, 32, 50, 0.14);
            --accent: #b42318;
            --accent-soft: rgba(180, 35, 24, 0.08);
            --success: #166534;
            --warning: #9a3412;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(180, 35, 24, 0.10), transparent 30%),
                linear-gradient(180deg, #f7f1e7 0%, #f2eadf 48%, #ebdfd0 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 4.75rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        p, div, label, span {
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Helvetica, sans-serif;
        }

        .hero-shell {
            padding: 0.25rem 0 1rem 0;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--line);
        }

        .eyebrow {
            display: inline-block;
            margin-bottom: 0.65rem;
            color: var(--accent);
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1;
            margin: 0;
            max-width: 12ch;
        }

        .hero-copy {
            color: var(--ink-soft);
            font-size: 0.96rem;
            max-width: 44rem;
            margin-top: 0.65rem;
            line-height: 1.5;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 0.95rem;
        }

        .status-chip {
            min-width: 12rem;
            padding: 0.8rem 0.95rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: var(--surface);
            backdrop-filter: blur(8px);
        }

        .status-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--ink-soft);
        }

        .status-value {
            margin-top: 0.35rem;
            font-size: 0.98rem;
            color: var(--ink);
            line-height: 1.35;
        }

        .section-title {
            font-size: 1.34rem;
            margin-bottom: 0.2rem;
        }

        .section-copy {
            color: var(--ink-soft);
            margin-bottom: 0.95rem;
            line-height: 1.55;
        }

        .selector-note {
            margin: 0.35rem 0 0.85rem 0;
            color: var(--ink-soft);
            font-size: 0.93rem;
        }

        .result-shell,
        .compare-card {
            background: linear-gradient(180deg, var(--surface-strong), var(--surface));
            border: 1px solid var(--line);
            border-radius: 24px;
            backdrop-filter: blur(10px);
        }

        .result-shell {
            padding: 1.2rem 1.2rem 1.1rem 1.2rem;
            min-height: 18rem;
        }

        .compare-card {
            position: relative;
            overflow: hidden;
            padding: 0.95rem;
            min-height: 11.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.72rem;
            transition:
                transform 160ms ease,
                border-color 160ms ease,
                box-shadow 160ms ease;
        }

        .compare-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: rgba(20, 32, 50, 0.18);
        }

        .compare-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 34px rgba(20, 32, 50, 0.08);
        }

        .compare-card.selected {
            border-color: rgba(180, 35, 24, 0.45);
            box-shadow: 0 0 0 1px rgba(180, 35, 24, 0.14) inset;
        }

        .compare-card.selected::before {
            background: linear-gradient(90deg, #b42318 0%, rgba(180, 35, 24, 0.34) 100%);
        }

        .compare-card.unavailable {
            background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,245,242,0.9));
        }

        .stTextInput > div > div > input,
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid var(--line);
            border-radius: 16px;
            color: var(--ink);
        }

        .stTextInput > label,
        .stTextArea > label,
        .stRadio > label {
            color: var(--ink);
            font-weight: 600;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.74);
            color: var(--ink);
            padding: 0.62rem 1rem;
            font-weight: 600;
        }

        .stButton > button[kind="primary"] {
            background: var(--ink);
            color: #fff;
            border-color: var(--ink);
        }

        .result-label,
        .compare-kicker,
        .meta-name,
        .term-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--ink-soft);
        }

        .compare-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(10.75rem, 1fr));
            gap: 0.8rem;
            margin-top: 0.72rem;
        }

        .compare-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--line);
        }

        .compare-summary {
            color: var(--ink-soft);
            font-size: 0.82rem;
            line-height: 1.35;
            text-align: right;
        }

        .result-risk {
            font-size: clamp(2rem, 4.7vw, 3rem);
            line-height: 1.02;
            margin-top: 0.45rem;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        }

        .result-prob,
        .compare-reason {
            font-size: 0.95rem;
            color: var(--ink-soft);
            line-height: 1.5;
        }

        .compare-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.65rem;
        }

        .compare-kicker {
            font-size: 0.66rem;
            letter-spacing: 0.12em;
            white-space: nowrap;
        }

        .compare-badge {
            border: 1px solid rgba(180, 35, 24, 0.22);
            border-radius: 999px;
            padding: 0.18rem 0.48rem;
            color: var(--accent);
            background: rgba(180, 35, 24, 0.07);
            font-size: 0.68rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .compare-name {
            font-size: clamp(1.08rem, 2.2vw, 1.28rem);
            color: var(--ink);
            line-height: 1.05;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .compare-status-row,
        .compare-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem;
        }

        .compare-pill,
        .compare-meta-pill {
            border: 1px solid rgba(20, 32, 50, 0.11);
            border-radius: 999px;
            padding: 0.24rem 0.5rem;
            background: rgba(255, 255, 255, 0.52);
            color: var(--ink-soft);
            font-size: 0.76rem;
            line-height: 1.1;
        }

        .compare-pill.danger {
            border-color: rgba(180, 35, 24, 0.20);
            background: rgba(180, 35, 24, 0.08);
            color: var(--accent);
        }

        .compare-pill.success {
            border-color: rgba(22, 101, 52, 0.18);
            background: rgba(22, 101, 52, 0.08);
            color: var(--success);
        }

        .compare-pill.warning {
            border-color: rgba(154, 52, 18, 0.18);
            background: rgba(154, 52, 18, 0.08);
            color: var(--warning);
        }

        .compare-score-block {
            margin-top: auto;
        }

        .compare-score {
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
            font-size: clamp(2rem, 4.8vw, 2.75rem);
            line-height: 0.9;
            color: var(--ink);
            letter-spacing: -0.04em;
        }

        .compare-score-label {
            margin-top: 0.2rem;
            color: var(--ink-soft);
            font-size: 0.82rem;
            line-height: 1.25;
        }

        .compare-reason {
            margin-top: 0.55rem;
            padding-top: 0.55rem;
            border-top: 1px dashed rgba(20, 32, 50, 0.14);
        }

        .prob-track {
            height: 12px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(20, 32, 50, 0.1);
            margin: 0.95rem 0 1rem 0;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #142032 0%, #b42318 100%);
        }

        .meta-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }

        .meta-unit {
            border-top: 1px solid var(--line);
            padding-top: 0.65rem;
        }

        .meta-value {
            margin-top: 0.32rem;
            font-size: 0.94rem;
            color: var(--ink);
            line-height: 1.35;
        }

        .detail-block,
        .term-block {
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--line);
        }

        .term-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.42rem 0;
            border-bottom: 1px dashed rgba(20, 32, 50, 0.12);
            font-size: 0.94rem;
        }

        .term-name {
            color: var(--ink);
            font-family: "SFMono-Regular", "Cascadia Code", Menlo, monospace;
        }

        .term-score {
            color: var(--ink-soft);
        }

        .info-note {
            padding: 1rem;
            border-left: 3px solid var(--accent);
            background: var(--accent-soft);
            color: var(--ink);
            border-radius: 16px;
            line-height: 1.6;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 4.25rem;
            }

            .meta-grid {
                grid-template-columns: 1fr 1fr;
            }

            .hero-title {
                max-width: 12ch;
            }

            .compare-summary {
                display: none;
            }
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 4rem;
            }

            .meta-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(model_states) -> None:
    loaded_count = sum(1 for state in model_states if state.is_loaded)
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="eyebrow">Fake Job Review Desk</div>
          <h1 class="hero-title">Two-model review</h1>
          <p class="hero-copy">
            Compare two English detectors on one posting, with Count LR as the default verdict.
          </p>
          <div class="status-row">
            <div class="status-chip">
              <div class="status-label">Models</div>
              <div class="status-value">Count LR, DistilBERT LR</div>
            </div>
            <div class="status-chip">
              <div class="status-label">Default Verdict</div>
              <div class="status-value">Count LR</div>
            </div>
            <div class="status-chip">
              <div class="status-label">Language Policy</div>
              <div class="status-value">English job text only</div>
            </div>
            <div class="status-chip">
              <div class="status-label">Artifacts</div>
              <div class="status-value">{loaded_count} / {len(UI_MODEL_SPECS)} bundles loaded</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel(model_states) -> bool:
    st.markdown('<div class="section-title">Input Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Pick the model you want in the primary verdict card, then run the same English posting through both available models.</div>',
        unsafe_allow_html=True,
    )

    st.radio(
        "Primary Result Model",
        options=[spec.model_id for spec in UI_MODEL_SPECS],
        format_func=lambda model_id: get_demo_model_spec(model_id).display_label,
        horizontal=True,
        key="selected_model_id",
    )
    st.markdown(
        '<div class="selector-note">The selected model owns the main verdict card. The compare strip still shows every model run.</div>',
        unsafe_allow_html=True,
    )

    st.radio(
        "Input Mode",
        options=[INPUT_MODE_COMBINED, INPUT_MODE_STRUCTURED],
        format_func=format_input_mode,
        horizontal=True,
        key="input_mode",
        on_change=reset_result_state,
    )

    sample_a, sample_b, sample_c = st.columns(3, gap="small")
    auto_predict = False
    if sample_a.button(SAMPLE_POSTS["legit"]["label"], use_container_width=True):
        apply_sample("legit")
        auto_predict = True
    if sample_b.button(SAMPLE_POSTS["suspicious"]["label"], use_container_width=True):
        apply_sample("suspicious")
        auto_predict = True
    if sample_c.button("Clear", use_container_width=True):
        clear_inputs()

    if st.session_state["input_mode"] == INPUT_MODE_COMBINED:
        st.text_area(
            "Combined Job Text",
            key="combined_text",
            height=380,
            placeholder="Paste one full English job posting here.",
        )
    else:
        render_structured_input_fields()

    unloaded = [state.spec.display_label for state in model_states if not state.is_loaded]
    if unloaded:
        st.warning(
            "Some model bundles are unavailable at load time: "
            + ", ".join(unloaded)
        )

    st.caption(
        "This UI compares the English Count LR and DistilBERT LR models only."
    )
    return st.button("Analyze Post", type="primary", use_container_width=True) or auto_predict


def render_structured_input_fields() -> None:
    for field in MODEL_TEXT_FIELDS:
        label = FIELD_LABELS[field]
        if field in {"title", "company_profile"}:
            st.text_input(label, key=field)
        else:
            st.text_area(label, key=field, height=150 if field == "description" else 120)


def render_result_panel(model_states, should_predict: bool) -> None:
    st.markdown('<div class="section-title">Result Panel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Read the selected verdict first, then compare both models and inspect explanations only where they are actually valid.</div>',
        unsafe_allow_html=True,
    )

    if should_predict:
        run_prediction(model_states)

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])
        return

    results = filter_ui_results(st.session_state["last_results"])
    if not results:
        st.markdown(
            """
            <div class="info-note">
              <strong>Demo flow</strong><br/>
              Start with one English sample to compare Count LR and DistilBERT LR on
              the same posting.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result_by_id = {result.model_id: result for result in results}
    selected_result = result_by_id[st.session_state["selected_model_id"]]
    contains_non_latin_input = bool(st.session_state.get("last_contains_non_latin", False))

    render_primary_result(selected_result, contains_non_latin_input)
    render_compare_surface(results, contains_non_latin_input)
    render_input_coverage(_pick_reference_prediction(results))
    render_selected_explanation(selected_result)


def render_primary_result(result, contains_non_latin_input: bool) -> None:
    spec = get_demo_model_spec(result.model_id)
    if result.status != "ready" or result.prediction is None:
        status_label = "Skipped" if result.status == "skipped" else "Unavailable"
        st.markdown(
            f"""
            <div class="result-shell">
              <div class="result-label">Selected Verdict</div>
              <div class="result-risk" style="color:#9a3412;">{status_label}</div>
              <div class="result-prob">The selected model stayed selected, but this run did not produce a score.</div>
              <div class="detail-block">
                <div class="meta-name">Model</div>
                <div class="meta-value">{escape(spec.display_label)}</div>
                <div class="meta-name" style="margin-top:0.8rem;">Type</div>
                <div class="meta-value">{escape(result.model_type)}</div>
                <div class="meta-name" style="margin-top:0.8rem;">Reason</div>
                <div class="meta-value">{escape(result.error_message or "Unknown runtime failure.")}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    prediction = result.prediction
    display_view = build_result_display_view(
        result,
        prediction,
        contains_non_latin_input=contains_non_latin_input,
    )
    probability = prediction.fraud_probability * 100
    risk_color = result_tone_color(display_view.status_tone)

    st.markdown(
        f"""
        <div class="result-shell">
          <div class="result-label">Selected Verdict</div>
          <div class="result-risk" style="color:{risk_color};">{escape(display_view.headline_label)}</div>
          <div class="result-prob">{escape(display_view.supporting_text)}</div>
          <div class="prob-track"><div class="prob-fill" style="width:{probability:.1f}%;"></div></div>
          <div class="meta-grid">
            <div class="meta-unit">
              <div class="meta-name">Model</div>
              <div class="meta-value">{escape(spec.display_label)}</div>
            </div>
            <div class="meta-unit">
              <div class="meta-name">Type</div>
              <div class="meta-value">{escape(prediction.model_type)}</div>
            </div>
            <div class="meta-unit">
              <div class="meta-name">Confidence</div>
              <div class="meta-value">{escape(display_view.confidence_label)}</div>
            </div>
            <div class="meta-unit">
              <div class="meta-name">Input Mode</div>
              <div class="meta-value">{escape(format_input_mode(st.session_state["last_input_mode"]))}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compare_surface(results, contains_non_latin_input: bool) -> None:
    ready_count = sum(
        1 for result in results if result.status == "ready" and result.prediction is not None
    )
    st.markdown(
        f"""
        <div class="compare-heading">
          <div class="term-title">Compare Surface</div>
          <div class="compare-summary">{ready_count} / {len(results)} models returned a score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_model_id = st.session_state["selected_model_id"]
    cards_html = "".join(
        render_compare_card(
            result,
            result.model_id == selected_model_id,
            contains_non_latin_input=contains_non_latin_input,
        )
        for result in results
    )
    st.markdown(
        f'<div class="compare-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def render_compare_card(result, is_selected: bool, contains_non_latin_input: bool) -> str:
    spec = get_demo_model_spec(result.model_id)
    classes = ["compare-card"]
    if is_selected:
        classes.append("selected")
    if result.status != "ready" or result.prediction is None:
        classes.append("unavailable")

    if result.status == "ready" and result.prediction is not None:
        prediction = result.prediction
        display_view = build_result_display_view(
            result,
            prediction,
            contains_non_latin_input=contains_non_latin_input,
        )
        probability_text = f"{prediction.fraud_probability * 100:.1f}%"
        probability_label = "fraud probability"
        verdict_text = display_view.headline_label
        status_text = "Ready"
        reason_html = ""
        type_text = format_compact_model_type(result.model_type)
        status_tone = display_view.status_tone
    else:
        probability_text = "--"
        probability_label = "no score"
        verdict_text = "Skipped" if result.status == "skipped" else "Unavailable"
        status_text = "Skipped" if result.status == "skipped" else "Unavailable"
        type_text = format_compact_model_type(result.model_type)
        status_tone = "warning"
        reason_html = (
            f'<div class="compare-reason">{escape(result.error_message or "Unknown runtime failure.")}</div>'
        )

    selected_tag = "Selected" if is_selected else "Compare"
    selected_badge = '<div class="compare-badge">Primary</div>' if is_selected else ""
    return (
        f'<div class="{" ".join(classes)}">'
        '<div class="compare-topline">'
        f'<div class="compare-kicker">{escape(selected_tag)}</div>'
        f"{selected_badge}"
        "</div>"
        f'<div class="compare-name" title="{escape(spec.display_label)}">{escape(spec.display_label)}</div>'
        '<div class="compare-status-row">'
        f'<div class="compare-pill">{escape(status_text)}</div>'
        f'<div class="compare-pill {escape(status_tone)}">{escape(verdict_text)}</div>'
        "</div>"
        '<div class="compare-score-block">'
        f'<div class="compare-score">{escape(probability_text)}</div>'
        f'<div class="compare-score-label">{escape(probability_label)}</div>'
        "</div>"
        '<div class="compare-meta-row">'
        f'<div class="compare-meta-pill">{escape(format_model_family(spec.family))}</div>'
        f'<div class="compare-meta-pill" title="{escape(result.model_type)}">{escape(type_text)}</div>'
        "</div>"
        f"{reason_html}"
        "</div>"
    )


def render_input_coverage(prediction: PredictionResult | None) -> None:
    if prediction is None:
        return
    st.markdown(
        f"""
        <div class="detail-block">
          <div class="term-title">Input Coverage</div>
          <div>{escape(format_active_fields(prediction.active_fields))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_explanation(result) -> None:
    spec = get_demo_model_spec(result.model_id)
    st.markdown(
        '<div class="detail-block"><div class="term-title">Selected Model Detail</div></div>',
        unsafe_allow_html=True,
    )

    if result.status != "ready" or result.prediction is None:
        message = "This selected model did not produce a score for the current run."
        st.markdown(
            f"""
            <div class="info-note">
              {escape(message)} Switch to one of the ready models in the compare surface for
              explanation detail.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    prediction = result.prediction
    if spec.family == MODEL_FAMILY_LEXICAL:
        positive_col, negative_col = st.columns(2, gap="large")
        with positive_col:
            render_term_rows(
                "Fraud signals",
                prediction.top_positive_terms,
                empty_text="No strong fraud-leaning terms matched this input.",
            )
        with negative_col:
            render_term_rows(
                "Legit signals",
                prediction.top_negative_terms,
                empty_text="No strong legitimacy-leaning terms matched this input.",
            )

        with st.expander("Processed text preview"):
            st.code(prediction.processed_text[:4000] or "(empty)", language="text")
        return

    if spec.family == MODEL_FAMILY_TRANSFORMER:
        st.markdown(
            """
            <div class="info-note">
              No lexical term explanation is shown for transformer-based models. Review the model
              input preview instead of fabricated token contributions.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Model input preview"):
            st.code(prediction.model_input_text[:4000] or "(empty)", language="text")


def render_term_rows(title: str, terms: tuple[TermContribution, ...], empty_text: str) -> None:
    st.markdown(f'<div class="term-title">{title}</div>', unsafe_allow_html=True)
    if not terms:
        st.caption(empty_text)
        return

    rows = []
    for term in terms:
        value_display = format_feature_value(term)
        rows.append(
            f"""
            <div class="term-row">
              <div class="term-name">{escape(term.term)}</div>
              <div class="term-score">{term.contribution:+.3f} · {escape(value_display)}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_model_notes(model_states) -> None:
    st.markdown("---")
    note_a, note_b, note_c = st.columns([1.08, 1, 1], gap="large")

    with note_a:
        st.subheader("Registry")
        st.write("The local web demo supports two English models.")
        for spec in UI_MODEL_SPECS:
            suffix = "default selected" if spec.model_id == UI_DEFAULT_MODEL_ID else "compare model"
            st.write(f"{spec.display_label}: {format_model_family(spec.family)} ({suffix})")

    with note_b:
        st.subheader("Runtime Contract")
        st.write("Count LR and DistilBERT LR are English-only demo models.")
        st.write("Chinese and other non-Latin input is blocked before prediction.")
        st.write("The selected model owns the primary verdict card.")
        st.write("The compare surface still renders every supported UI model run.")

    with note_c:
        st.subheader("Artifacts")
        for state in model_states:
            status = "loaded" if state.is_loaded else "missing"
            st.write(f"{state.spec.display_label}: `{Path(state.spec.artifact_dir).as_posix()}`")
            st.caption(f"Status: {status}" + (f" · {state.error_message}" if state.error_message else ""))


def run_prediction(model_states) -> None:
    reset_result_state()
    input_mode = st.session_state["input_mode"]
    st.session_state["last_input_mode"] = input_mode

    if input_mode == INPUT_MODE_COMBINED:
        combined_text = str(st.session_state.get("combined_text", "") or "").strip()
        if not combined_text:
            st.session_state["last_error"] = "Please enter combined job text."
            return
        contains_non_latin = contains_unsupported_script(combined_text)
        st.session_state["last_contains_non_latin"] = contains_non_latin
        if contains_non_latin:
            st.session_state["last_error"] = "This UI supports English job text only. Please remove Chinese or other non-Latin text."
            return
        st.session_state["last_results"] = run_demo_models_raw_text(combined_text, model_states)
        return

    job_post = build_job_post_from_state()
    combined_text = "\n".join(value for value in job_post.values() if value)
    if not combined_text.strip():
        st.session_state["last_error"] = "Please fill at least one structured field."
        return
    contains_non_latin = contains_unsupported_script(combined_text)
    st.session_state["last_contains_non_latin"] = contains_non_latin
    if contains_non_latin:
        st.session_state["last_error"] = "This UI supports English job text only. Please remove Chinese or other non-Latin text."
        return
    st.session_state["last_results"] = run_demo_models_job_post(job_post, model_states)


def build_job_post_from_state() -> dict[str, str]:
    return {
        field: str(st.session_state.get(field, "") or "").strip()
        for field in MODEL_TEXT_FIELDS
    }


def build_demo_combined_text(job_post: dict[str, str]) -> str:
    return "\n\n".join(value.strip() for value in job_post.values() if value.strip())


def contains_unsupported_script(text: str) -> bool:
    return bool(NON_LATIN_SCRIPT_RE.search(text))


def apply_sample(name: str) -> None:
    sample_job_post = SAMPLE_POSTS[name]["job_post"]
    for field in MODEL_TEXT_FIELDS:
        st.session_state[field] = sample_job_post.get(field, "")
    st.session_state["combined_text"] = build_demo_combined_text(sample_job_post)
    reset_result_state()


def clear_inputs() -> None:
    st.session_state["combined_text"] = ""
    for field in MODEL_TEXT_FIELDS:
        st.session_state[field] = ""
    reset_result_state()


def format_active_fields(fields: tuple[str, ...]) -> str:
    if not fields:
        return "None"
    if fields == ("combined_text",):
        return "Combined job text"
    return ", ".join(FIELD_LABELS.get(field, field) for field in fields)


def format_feature_value(term: TermContribution) -> str:
    if term.feature_kind == "count":
        return f"count {int(round(term.feature_value))}"
    return f"tfidf {term.feature_value:.3f}"


def format_input_mode(mode: str) -> str:
    if mode == INPUT_MODE_STRUCTURED:
        return "Structured fields"
    return "Combined text"


def format_model_family(family: str) -> str:
    if family == MODEL_FAMILY_TRANSFORMER:
        return "Transformer"
    return "Lexical"


def format_compact_model_type(model_type: str) -> str:
    compact_labels = {
        "CountVectorizer + LogisticRegression": "Count + LR",
        "DistilBERT Embedding + LogisticRegression": "DistilBERT + LR",
    }
    return compact_labels.get(model_type, model_type)


def build_result_display_view(
    result,
    prediction: PredictionResult,
    *,
    contains_non_latin_input: bool,
) -> ResultDisplayView:
    probability_text = f"{prediction.fraud_probability * 100:.1f}% fraud probability"
    return ResultDisplayView(
        headline_label=prediction.risk_label,
        supporting_text=f"{probability_text} from {result.display_label}.",
        confidence_label=prediction.confidence_band,
        status_tone="danger" if prediction.label == 1 else "success",
    )


def result_tone_color(status_tone: str) -> str:
    if status_tone == "danger":
        return "#b42318"
    if status_tone == "warning":
        return "#9a3412"
    return "#166534"


def _pick_reference_prediction(results) -> PredictionResult | None:
    for result in results:
        if result.status == "ready" and result.prediction is not None:
            return result.prediction
    return None


def filter_ui_results(results) -> tuple:
    if not results:
        return ()
    return tuple(result for result in results if result.model_id in UI_MODEL_IDS)


if __name__ == "__main__":
    main()
