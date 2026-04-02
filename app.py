from __future__ import annotations

from dataclasses import dataclass
import re
from html import escape
from pathlib import Path

import streamlit as st

from fake_job_demo import (
    DEFAULT_SELECTED_MODEL_ID,
    DEMO_MODEL_SPECS,
    FIELD_LABELS,
    MODEL_FAMILY_LEXICAL,
    MODEL_FAMILY_MULTILINGUAL,
    MODEL_FAMILY_TRANSFORMER,
    MODEL_TEXT_FIELDS,
    MULTILINGUAL_PRIMARY_MODEL_ID,
    PredictionResult,
    TermContribution,
    get_demo_model_spec,
    load_demo_model_states,
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
    "zh_legit": {
        "label": "ZH Legit",
        "job_post": {
            "title": "数据分析师",
            "company_profile": "成熟的医疗数据服务公司，为医院和保险机构提供分析支持。",
            "description": "负责运营报表、数据质量排查，并与产品和业务团队一起优化指标定义。",
            "requirements": "具备 SQL 和 Python 分析经验，沟通清晰，能够向非技术同事汇报结果。",
            "benefits": "五险一金，年度体检，培训补贴，混合办公。",
        },
    },
    "zh_suspicious": {
        "label": "ZH Suspicious",
        "job_post": {
            "title": "居家兼职录入员",
            "company_profile": "国际金融咨询团队，快速扩张中。",
            "description": "无需经验，手机即可上岗，日结高薪，立即联系招聘顾问开始赚钱。",
            "requirements": "需保持在线，能使用 Telegram，每天提交简单报表。",
            "benefits": "高额提成，快速返现，推荐奖金。",
        },
    },
}


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
    st.session_state.setdefault("selected_model_id", DEFAULT_SELECTED_MODEL_ID)
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
    return load_demo_model_states()


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
            padding-top: 1.8rem;
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
            padding: 0 0 1.25rem 0;
            margin-bottom: 1.25rem;
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
            font-size: clamp(2.2rem, 5vw, 3.6rem);
            line-height: 0.95;
            margin: 0;
            max-width: 11ch;
        }

        .hero-copy {
            color: var(--ink-soft);
            font-size: 1rem;
            max-width: 56rem;
            margin-top: 0.85rem;
            line-height: 1.65;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 1.1rem;
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
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 16rem;
        }

        .compare-card.selected {
            border-color: rgba(180, 35, 24, 0.45);
            box-shadow: 0 0 0 1px rgba(180, 35, 24, 0.14) inset;
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

        .result-risk {
            font-size: clamp(2rem, 4.7vw, 3rem);
            line-height: 1.02;
            margin-top: 0.45rem;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        }

        .result-prob,
        .compare-prob,
        .compare-meta,
        .compare-reason {
            font-size: 0.95rem;
            color: var(--ink-soft);
            line-height: 1.5;
        }

        .compare-name {
            font-size: 1.2rem;
            color: var(--ink);
            margin-top: 0.4rem;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        }

        .compare-status {
            font-size: 1.05rem;
            color: var(--ink);
            margin-top: 0.45rem;
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
            .meta-grid {
                grid-template-columns: 1fr 1fr;
            }

            .hero-title {
                max-width: 12ch;
            }
        }

        @media (max-width: 640px) {
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
          <h1 class="hero-title">Four models, one selected verdict, multilingual coverage when you need it.</h1>
          <p class="hero-copy">
            This local demo compares Count LR, TF-IDF LR, DistilBERT LR, and a restored
            multilingual primary model on the same posting. TF-IDF stays the default selected
            verdict for English input, while the multilingual model can keep scoring when the input
            includes non-Latin scripts.
          </p>
          <div class="status-row">
            <div class="status-chip">
              <div class="status-label">Models</div>
              <div class="status-value">Count LR, TF-IDF LR, DistilBERT LR, Multilingual Primary</div>
            </div>
            <div class="status-chip">
              <div class="status-label">Default Verdict</div>
              <div class="status-value">TF-IDF LR</div>
            </div>
            <div class="status-chip">
              <div class="status-label">Language Policy</div>
              <div class="status-value">English models plus multilingual fallback</div>
            </div>
            <div class="status-chip">
              <div class="status-label">Artifacts</div>
              <div class="status-value">{loaded_count} / {len(DEMO_MODEL_SPECS)} bundles loaded</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel(model_states) -> bool:
    st.markdown('<div class="section-title">Input Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Pick the model you want in the primary verdict card, then run the same posting through every available model. English-only models are skipped automatically when the input contains non-Latin scripts.</div>',
        unsafe_allow_html=True,
    )

    st.radio(
        "Primary Result Model",
        options=[spec.model_id for spec in DEMO_MODEL_SPECS],
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

    sample_a, sample_b, sample_c, sample_d, sample_e = st.columns(5, gap="small")
    auto_predict = False
    if sample_a.button(SAMPLE_POSTS["legit"]["label"], use_container_width=True):
        apply_sample("legit")
        auto_predict = True
    if sample_b.button(SAMPLE_POSTS["suspicious"]["label"], use_container_width=True):
        apply_sample("suspicious")
        auto_predict = True
    if sample_c.button(SAMPLE_POSTS["zh_legit"]["label"], use_container_width=True):
        apply_sample("zh_legit")
        auto_predict = True
    if sample_d.button(SAMPLE_POSTS["zh_suspicious"]["label"], use_container_width=True):
        apply_sample("zh_suspicious")
        auto_predict = True
    if sample_e.button("Clear", use_container_width=True):
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
        "English lexical models and DistilBERT skip non-Latin input. Multilingual Primary handles multilingual text when its bundle is available."
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
        '<div class="section-copy">Read the selected verdict first, then compare all four models and inspect explanations only where they are actually valid.</div>',
        unsafe_allow_html=True,
    )

    if should_predict:
        run_prediction(model_states)

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])
        return

    if not st.session_state["last_results"]:
        st.markdown(
            """
            <div class="info-note">
              <strong>Demo flow</strong><br/>
              Start with one English sample to compare all four models, then trigger one of the ZH
              samples to show the multilingual path take over while the English-only models mark
              themselves as skipped.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    results = tuple(st.session_state["last_results"])
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
    st.markdown('<div class="detail-block"><div class="term-title">Compare Surface</div></div>', unsafe_allow_html=True)
    selected_model_id = st.session_state["selected_model_id"]
    cards_per_row = 2 if len(results) > 3 else len(results)

    for start in range(0, len(results), cards_per_row):
        row_results = results[start : start + cards_per_row]
        columns = st.columns(cards_per_row, gap="large")
        for column, result in zip(columns, row_results):
            with column:
                render_compare_card(
                    result,
                    result.model_id == selected_model_id,
                    contains_non_latin_input=contains_non_latin_input,
                )


def render_compare_card(result, is_selected: bool, contains_non_latin_input: bool) -> None:
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
        probability_text = display_view.supporting_text
        verdict_text = display_view.headline_label
        status_text = "Ready"
        reason_html = ""
        type_text = result.model_type
        status_color = result_tone_color(display_view.status_tone)
    else:
        probability_text = "No probability available"
        verdict_text = "Skipped" if result.status == "skipped" else "Unavailable"
        status_text = "Skipped" if result.status == "skipped" else "Unavailable"
        type_text = result.model_type
        status_color = "#142032"
        reason_html = (
            f'<div class="compare-reason">{escape(result.error_message or "Unknown runtime failure.")}</div>'
        )

    selected_tag = "Selected verdict" if is_selected else "Compare model"
    st.markdown(
        f"""
        <div class="{' '.join(classes)}">
          <div class="compare-kicker">{escape(selected_tag)}</div>
          <div class="compare-name">{escape(spec.display_label)}</div>
          <div class="compare-status" style="color:{status_color};">{escape(status_text)} · {escape(verdict_text)}</div>
          <div class="compare-prob">{escape(probability_text)}</div>
          <div class="compare-meta">Family: {escape(format_model_family(spec.family))}</div>
          <div class="compare-meta">Type: {escape(type_text)}</div>
          {reason_html}
        </div>
        """,
        unsafe_allow_html=True,
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
        message = (
            "This selected model was skipped because the current input contains non-Latin text and the model is English-only."
            if result.status == "skipped"
            else "This selected model did not produce a score for the current run."
        )
        st.markdown(
            f"""
            <div class="info-note">
              {escape(message)} Keep the selector on it if you want to demonstrate routing behavior,
              or switch to one of the ready models in the compare surface for explanation detail.
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

    if spec.family in {MODEL_FAMILY_TRANSFORMER, MODEL_FAMILY_MULTILINGUAL}:
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
        st.write("The local web demo supports four models.")
        for spec in DEMO_MODEL_SPECS:
            suffix = "default selected" if spec.model_id == DEFAULT_SELECTED_MODEL_ID else "compare model"
            st.write(f"{spec.display_label}: {format_model_family(spec.family)} ({suffix})")

    with note_b:
        st.subheader("Runtime Contract")
        st.write("Count LR, TF-IDF LR, and DistilBERT LR are English-first models.")
        st.write("Multilingual Primary accepts multilingual input and remains active for non-Latin text.")
        st.write("English-only models are shown as skipped instead of returning misleading scores on non-Latin input.")
        st.write("Low-score non-Latin results from Multilingual Primary are surfaced as Needs Review instead of Legit.")
        st.write("If the local mBERT backbone crashes at runtime, Multilingual Primary is shown as unavailable with the subprocess reason.")

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
        if contains_non_latin and not has_multilingual_bundle(model_states):
            st.session_state["last_error"] = (
                "Non-Latin input requires the multilingual_primary bundle, but it is not available."
            )
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
    if contains_non_latin and not has_multilingual_bundle(model_states):
        st.session_state["last_error"] = (
            "Non-Latin input requires the multilingual_primary bundle, but it is not available."
        )
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
    if family == MODEL_FAMILY_MULTILINGUAL:
        return "Multilingual transformer"
    if family == MODEL_FAMILY_TRANSFORMER:
        return "Transformer embedding"
    return "Lexical"


def build_result_display_view(
    result,
    prediction: PredictionResult,
    *,
    contains_non_latin_input: bool,
) -> ResultDisplayView:
    probability_text = f"{prediction.fraud_probability * 100:.1f}% fraud probability"
    if result.model_id == MULTILINGUAL_PRIMARY_MODEL_ID and contains_non_latin_input:
        headline_label = (
            "Suspicious"
            if prediction.fraud_probability >= prediction.threshold
            else "Needs Review"
        )
        status_tone = "danger" if headline_label == "Suspicious" else "warning"
        return ResultDisplayView(
            headline_label=headline_label,
            supporting_text=f"Experimental multilingual score: {probability_text}.",
            confidence_label="Experimental",
            status_tone=status_tone,
        )

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


def has_multilingual_bundle(model_states) -> bool:
    for state in model_states:
        if state.spec.model_id == MULTILINGUAL_PRIMARY_MODEL_ID and state.is_loaded:
            return True
    return False


if __name__ == "__main__":
    main()
