from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from fake_job_demo import (
    BASELINE_MODEL_ID,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MULTILINGUAL_ARTIFACT_DIR,
    PRIMARY_MODEL_ID,
    build_raw_text_from_fields,
    load_model_bundle_state,
    predict_demo_raw_text,
)

st.set_page_config(
    page_title="Fake Job Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SAMPLE_POSTS = {
    "legit": {
        "title": "Senior Data Analyst",
        "company_profile": (
            "Established healthcare analytics company supporting hospitals and payer operations."
        ),
        "description": (
            "Build dashboards, validate data quality, and partner with finance and operations teams "
            "to improve reporting accuracy across recurring monthly close workflows."
        ),
        "requirements": (
            "3+ years of SQL experience, Python for analysis, stakeholder communication, and "
            "experience working with BI tools in a regulated environment."
        ),
        "benefits": (
            "Medical, dental, retirement plan, professional development budget, and hybrid work policy."
        ),
    },
    "suspicious": {
        "title": "Data Entry Receptionist",
        "company_profile": "Online financing office",
        "description": (
            "Earn money from home with cash wages. Apply today for data entry, "
            "receptionist, and clerical processing work."
        ),
        "requirements": (
            "Type fast, motivated, clerical aptitude, accountant mindset, and send daily reports."
        ),
        "benefits": "Weekly cash wages and same-day payment.",
    },
}


def main() -> None:
    inject_styles()
    initialize_state()
    model_state = load_runtime_state()

    render_hero(model_state)

    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    auto_predict = False

    with left_col:
        auto_predict = render_input_panel(model_state)
        if model_state.baseline_error and model_state.baseline_bundle is None:
            render_missing_baseline_artifacts(model_state.baseline_error)

    with right_col:
        render_result_panel(model_state, auto_predict)

    render_model_notes(model_state)


def initialize_state() -> None:
    st.session_state.setdefault("combined_text", "")
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", "")


def load_runtime_state():
    return load_model_bundle_state(
        DEFAULT_ARTIFACT_DIR,
        DEFAULT_MULTILINGUAL_ARTIFACT_DIR,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f3ede3;
            --paper-deep: #e5dccd;
            --ink: #182033;
            --ink-soft: #4b566f;
            --accent: #b42318;
            --accent-soft: rgba(180, 35, 24, 0.08);
            --line: rgba(24, 32, 51, 0.14);
            --success: #166534;
            --surface: rgba(255, 255, 255, 0.55);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(180, 35, 24, 0.12), transparent 28%),
                linear-gradient(180deg, #f6f1e8 0%, #f1eadf 52%, #ece2d4 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.25rem;
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
            padding: 0 0 1.5rem 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.75rem;
        }

        .eyebrow {
            display: inline-block;
            margin-bottom: 0.65rem;
            color: var(--accent);
            font-size: 0.82rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: clamp(2.75rem, 7vw, 5.1rem);
            line-height: 0.95;
            margin: 0;
            max-width: 9ch;
        }

        .hero-copy {
            color: var(--ink-soft);
            font-size: 1.04rem;
            max-width: 56rem;
            margin-top: 1rem;
            line-height: 1.7;
        }

        .signal-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.3rem;
        }

        .signal-unit {
            border-top: 1px solid var(--line);
            padding-top: 0.75rem;
        }

        .signal-label {
            color: var(--ink-soft);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .signal-value {
            margin-top: 0.35rem;
            font-size: 1.05rem;
            color: var(--ink);
        }

        .panel-title {
            font-size: 1.45rem;
            margin-bottom: 0.25rem;
        }

        .panel-copy {
            color: var(--ink-soft);
            margin-bottom: 1rem;
            line-height: 1.6;
        }

        .stTextInput > div > div > input,
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.66);
            border: 1px solid var(--line);
            border-radius: 16px;
            color: var(--ink);
        }

        .stTextInput > label,
        .stTextArea > label {
            color: var(--ink);
            font-weight: 600;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.66);
            color: var(--ink);
            padding: 0.65rem 1rem;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: rgba(24, 32, 51, 0.28);
            color: var(--ink);
        }

        .stButton > button[kind="primary"] {
            background: var(--ink);
            color: #fff;
            border-color: var(--ink);
        }

        .result-shell {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.56), rgba(255, 255, 255, 0.38));
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.35rem 1.35rem 1.2rem 1.35rem;
            min-height: 22rem;
            backdrop-filter: blur(10px);
        }

        .result-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--ink-soft);
        }

        .result-risk {
            font-size: clamp(2rem, 5vw, 3rem);
            line-height: 1.02;
            margin-top: 0.4rem;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        }

        .result-prob {
            font-size: 1rem;
            color: var(--ink-soft);
            margin-top: 0.45rem;
        }

        .prob-track {
            height: 12px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(24, 32, 51, 0.1);
            margin: 1rem 0 1rem 0;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #182033 0%, #b42318 100%);
        }

        .meta-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1.1rem 0 1.35rem 0;
        }

        .meta-unit {
            border-top: 1px solid var(--line);
            padding-top: 0.7rem;
        }

        .meta-unit .meta-name {
            color: var(--ink-soft);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .meta-unit .meta-value {
            margin-top: 0.35rem;
            font-size: 0.98rem;
            color: var(--ink);
        }

        .compare-title {
            font-size: 0.9rem;
            color: var(--ink-soft);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.55rem;
        }

        .compare-card {
            background: rgba(255, 255, 255, 0.54);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 10rem;
        }

        .compare-card.is-active {
            border-color: rgba(180, 35, 24, 0.34);
            box-shadow: inset 0 0 0 1px rgba(180, 35, 24, 0.12);
        }

        .compare-name {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--ink-soft);
        }

        .compare-value {
            margin-top: 0.45rem;
            font-size: 1.45rem;
            color: var(--ink);
            line-height: 1.1;
        }

        .compare-prob {
            margin-top: 0.45rem;
            font-size: 0.98rem;
            color: var(--ink-soft);
        }

        .compare-meta {
            margin-top: 0.75rem;
            font-size: 0.86rem;
            color: var(--ink-soft);
            line-height: 1.5;
        }

        .term-block {
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--line);
        }

        .term-title {
            font-size: 0.9rem;
            color: var(--ink-soft);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.55rem;
        }

        .term-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.45rem 0;
            border-bottom: 1px dashed rgba(24, 32, 51, 0.12);
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
            padding: 1rem 1rem 1rem 1rem;
            border-left: 3px solid var(--accent);
            background: var(--accent-soft);
            color: var(--ink);
            border-radius: 16px;
        }

        .subtle {
            color: var(--ink-soft);
        }

        @media (max-width: 900px) {
            .signal-strip,
            .meta-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                max-width: 12ch;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(model_state) -> None:
    routing_value = (
        "Multilingual primary + lexical baseline"
        if model_state.multilingual_bundle
        else "Baseline only until Kaggle export is installed"
    )
    artifact_gate = (
        "model/multilingual_primary is ready"
        if model_state.multilingual_bundle
        else "Drop Kaggle export into model/multilingual_primary"
    )
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="eyebrow">Fake Job Screening Demo</div>
          <h1 class="hero-title">Route to the stronger model when it exists.</h1>
          <p class="hero-copy">
            This demo now uses a gated path: the saved lexical baseline always remains available,
            and a multilingual transformer head becomes the default only after Kaggle exports the
            classifier bundle into <code>model/multilingual_primary</code>. 中文输入流程可跑，但效果仍取决于导出的多语言工件。
          </p>
          <div class="signal-strip">
            <div class="signal-unit">
              <div class="signal-label">Routing</div>
              <div class="signal-value">{escape(routing_value)}</div>
            </div>
            <div class="signal-unit">
              <div class="signal-label">Baseline</div>
              <div class="signal-value">CountVectorizer + LogisticRegression</div>
            </div>
            <div class="signal-unit">
              <div class="signal-label">Artifact Gate</div>
              <div class="signal-value">{escape(artifact_gate)}</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel(model_state) -> bool:
    st.markdown('<div class="panel-title">Job Post Input / 招聘信息输入</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Paste one complete job posting. The app will try the multilingual primary model if its Kaggle export is installed, and always keep the lexical baseline for comparison when available.</div>',
        unsafe_allow_html=True,
    )

    sample_a, sample_b, sample_c = st.columns([1, 1, 1], gap="small")
    auto_predict = False
    if sample_a.button("Load legit sample / 真实样例", use_container_width=True):
        apply_sample("legit")
        auto_predict = True
    if sample_b.button("Load suspicious sample / 可疑样例", use_container_width=True):
        apply_sample("suspicious")
        auto_predict = True
    if sample_c.button("Clear / 清空", use_container_width=True):
        clear_inputs()

    st.text_area(
        "Combined Job Text / 合并整段招聘文本",
        key="combined_text",
        height=420,
        placeholder="Paste the entire job post here if you copied one long block from CSV, Kaggle, or another site.",
    )
    st.caption("Recommended demo mode: paste the full posting text in one block.")
    if model_state.multilingual_bundle is None:
        st.caption(
            "当前仅 baseline。先在 Kaggle 导出多语言工件，再放入 model/multilingual_primary，web 才会切换到主模型。"
        )
    else:
        model_name = model_state.multilingual_bundle.hf_model_name
        st.caption(f"Multilingual primary is installed and will load `{model_name}` on first use.")

    primary_run, secondary_run = st.columns([1, 1.1], gap="small")
    analyze_clicked = primary_run.button(
        "Analyze Post / 运行判断",
        type="primary",
        use_container_width=True,
    )
    secondary_run.markdown(
        '<p class="subtle" style="padding-top:0.55rem;">The active result may come from the multilingual model, while lexical cue rows below remain sourced from the baseline.</p>',
        unsafe_allow_html=True,
    )
    return auto_predict or analyze_clicked


def render_result_panel(model_state, should_predict: bool) -> None:
    st.markdown('<div class="panel-title">Prediction Desk / 结果面板</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Read the active score first, then inspect the side-by-side model outputs and the baseline lexical cues.</div>',
        unsafe_allow_html=True,
    )

    if model_state.multilingual_bundle is None:
        st.info(
            "Multilingual artifacts are not installed yet. The app will keep using the baseline until Kaggle exports are placed under model/multilingual_primary."
        )

    if should_predict:
        combined_text = st.session_state.get("combined_text", "").strip()
        if combined_text:
            try:
                st.session_state.last_result = predict_demo_raw_text(combined_text, model_state)
                st.session_state.last_error = ""
            except Exception as exc:
                st.session_state.last_result = None
                st.session_state.last_error = str(exc)
        else:
            st.session_state.last_result = None
            st.session_state.last_error = "Please enter combined job text. / 请填写整段招聘文本。"

    if st.session_state.last_error:
        st.error(st.session_state.last_error)
        return

    if st.session_state.last_result is None:
        st.markdown(
            """
            <div class="info-note">
              <strong>How to use / 使用方式</strong><br/>
              Load one of the two curated samples or paste your own job posting. The app will try the
              multilingual Kaggle export first when available, and otherwise keep scoring with the saved
              baseline artifact under <code>model/mvp</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result = st.session_state.last_result
    active_result = result.active_result
    if active_result is None:
        st.error("No active prediction result is available.")
        return

    risk_color = "#b42318" if active_result.label == 1 else "#166534"
    probability = active_result.fraud_probability * 100
    if result.primary_result is not None and result.baseline_result is not None:
        compare_label = "Primary + Baseline"
    elif result.primary_result is not None:
        compare_label = "Primary only"
    else:
        compare_label = "Baseline only"
    st.markdown(
        f"""
        <div class="result-shell">
          <div class="result-label">Active Model / 当前生效模型</div>
          <div class="result-risk" style="color:{risk_color};">{escape(active_result.risk_label)}</div>
          <div class="result-prob">{probability:.1f}% fraud probability from {escape(active_result.model_label)}.</div>
          <div class="prob-track"><div class="prob-fill" style="width:{probability:.1f}%;"></div></div>
          <div class="meta-grid">
            <div class="meta-unit">
              <div class="meta-name">Model</div>
              <div class="meta-value">{escape(active_result.model_label)}</div>
            </div>
            <div class="meta-unit">
              <div class="meta-name">Threshold</div>
              <div class="meta-value">{active_result.threshold:.2f}</div>
            </div>
            <div class="meta-unit">
              <div class="meta-name">Compare Mode</div>
              <div class="meta-value">{escape(compare_label)}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.fallback_reason and active_result.model_id == BASELINE_MODEL_ID:
        st.warning(result.fallback_reason)
    elif result.fallback_reason:
        st.caption(result.fallback_reason)

    st.markdown('<div class="compare-title">Model Compare / 模型对照</div>', unsafe_allow_html=True)
    compare_a, compare_b = st.columns(2, gap="large")
    with compare_a:
        render_model_card(
            title="Primary multilingual / 多语言主模型",
            prediction=result.primary_result,
            is_active=result.active_model_id == PRIMARY_MODEL_ID,
            empty_text="Install Kaggle export in model/multilingual_primary to activate this card.",
        )
    with compare_b:
        render_model_card(
            title="Baseline lexical / 词袋基线",
            prediction=result.baseline_result,
            is_active=result.active_model_id == BASELINE_MODEL_ID,
            empty_text="Baseline artifacts are missing from model/mvp.",
        )

    st.markdown(
        f"""
        <div class="term-block">
          <div class="term-title">Input Coverage / 输入覆盖字段</div>
          <div>{escape(format_active_fields(active_result.active_fields))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.baseline_result is not None:
        if result.active_model_id != BASELINE_MODEL_ID:
            st.caption(
                "Lexical cue rows below come from the baseline model for transparency. The active score above still comes from the multilingual primary model."
            )
        positive_col, negative_col = st.columns(2, gap="large")
        with positive_col:
            render_term_rows(
                "Fraud signals / 推高可疑概率的词",
                result.baseline_result.top_positive_terms,
                empty_text="No strong fraud-leaning baseline terms matched this input.",
            )
        with negative_col:
            render_term_rows(
                "Legit signals / 拉低可疑概率的词",
                result.baseline_result.top_negative_terms,
                empty_text="No strong legitimacy-leaning baseline terms matched this input.",
            )

        with st.expander("Baseline processed text preview / 基线预处理预览"):
            st.code(result.baseline_result.processed_text[:4000] or "(empty)", language="text")

    if result.primary_result is not None:
        with st.expander("Primary model input preview / 主模型输入预览"):
            st.code(result.primary_result.model_input_text[:4000] or "(empty)", language="text")


def render_model_card(title: str, prediction, is_active: bool, empty_text: str) -> None:
    if prediction is None:
        st.markdown(
            f"""
            <div class="compare-card">
              <div class="compare-name">{escape(title)}</div>
              <div class="compare-value">Unavailable</div>
              <div class="compare-meta">{escape(empty_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    probability = prediction.fraud_probability * 100
    active_class = " is-active" if is_active else ""
    st.markdown(
        f"""
        <div class="compare-card{active_class}">
          <div class="compare-name">{escape(title)}</div>
          <div class="compare-value">{escape(prediction.risk_label)}</div>
          <div class="compare-prob">{probability:.1f}% fraud probability</div>
          <div class="compare-meta">
            Threshold {prediction.threshold:.2f}<br/>
            {escape(prediction.model_type)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_term_rows(title: str, terms, empty_text: str) -> None:
    st.markdown(f'<div class="term-title">{title}</div>', unsafe_allow_html=True)
    if not terms:
        st.caption(empty_text)
        return

    rows = []
    for term in terms:
        rows.append(
            f"""
            <div class="term-row">
              <div class="term-name">{escape(term.term)}</div>
              <div class="term-score">{term.contribution:+.3f} · x{term.count}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_model_notes(model_state) -> None:
    st.markdown("---")
    note_a, note_b, note_c = st.columns([1.05, 1, 1], gap="large")

    with note_a:
        st.subheader("Model Notes / 模型说明")
        st.write(
            "This app now has a gated routing layer. It does not assume multilingual artifacts exist."
        )
        st.write(
            "If `model/multilingual_primary` contains a Kaggle-exported classifier bundle, the multilingual model becomes the active prediction path."
        )
        st.write(
            "If those artifacts are missing or unreadable, the app falls back to the saved lexical baseline under `model/mvp`."
        )

    with note_b:
        st.subheader("Preprocess / 预处理")
        if model_state.baseline_bundle:
            steps = model_state.baseline_bundle.metadata.get("preprocess", [])
            st.write("Baseline: " + " -> ".join(steps))
        else:
            st.write("Baseline artifacts not loaded.")

        if model_state.multilingual_bundle:
            preprocess_mode = model_state.multilingual_bundle.metadata.get("preprocess_mode", "")
            hf_model_name = model_state.multilingual_bundle.hf_model_name
            st.write(f"Primary: {preprocess_mode}")
            st.write(f"Backbone: `{hf_model_name}`")
        else:
            st.write("Primary: waiting for Kaggle export.")

    with note_c:
        st.subheader("Artifacts / 模型产物")
        st.write(f"Baseline path: `{Path(DEFAULT_ARTIFACT_DIR).as_posix()}`")
        st.write(f"Primary path: `{Path(DEFAULT_MULTILINGUAL_ARTIFACT_DIR).as_posix()}`")
        st.write("Expected Kaggle export: `classifier.joblib`, `metadata.json`, `cv_results.csv`, `README.txt`")
        if model_state.multilingual_error:
            st.caption(model_state.multilingual_error)


def render_missing_baseline_artifacts(error_message: str) -> None:
    st.warning("Baseline artifacts are missing or unreadable.")
    st.code(
        "\n".join(
            [
                "Expected files under model/mvp/:",
                "- count_vectorizer.joblib",
                "- logreg_model.joblib",
                "- metadata.json",
                "",
                "Then run:",
                "pip install -r requirements.txt",
                "streamlit run app.py",
            ]
        ),
        language="text",
    )
    st.caption(error_message)


def apply_sample(name: str) -> None:
    st.session_state["combined_text"] = build_raw_text_from_fields(**SAMPLE_POSTS[name])
    st.session_state.last_result = None
    st.session_state.last_error = ""


def clear_inputs() -> None:
    st.session_state["combined_text"] = ""
    st.session_state.last_result = None
    st.session_state.last_error = ""


def format_active_fields(fields: tuple[str, ...]) -> str:
    if not fields:
        return "None"
    if fields == ("combined_text",):
        return "combined_text / 合并整段文本"
    return ", ".join(fields)


if __name__ == "__main__":
    main()
