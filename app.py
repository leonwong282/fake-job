from __future__ import annotations

from pathlib import Path

import streamlit as st

from fake_job_demo import DEFAULT_ARTIFACT_DIR, build_raw_text_from_fields, load_artifacts, predict_raw_text

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
    bundle, bundle_error = load_bundle()

    render_hero()

    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    auto_predict = False

    with left_col:
        auto_predict = render_input_panel()
        if bundle_error:
            render_missing_artifacts(bundle_error)

    with right_col:
        render_result_panel(bundle, bundle_error, auto_predict)

    render_model_notes(bundle)


def initialize_state() -> None:
    st.session_state.setdefault("combined_text", "")
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", "")


def load_bundle():
    try:
        return load_artifacts(DEFAULT_ARTIFACT_DIR), ""
    except Exception as exc:
        return None, str(exc)


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
            max-width: 8ch;
        }

        .hero-copy {
            color: var(--ink-soft);
            font-size: 1.04rem;
            max-width: 52rem;
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
            min-height: 28rem;
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


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
          <div class="eyebrow">Fake Job Screening Demo</div>
          <h1 class="hero-title">Spot the signal before the scam lands.</h1>
          <p class="hero-copy">
            A bilingual demo for the saved Kaggle MVP model. Enter a job post in structured fields,
            then inspect the model's fraud probability and the terms pulling the score up or down.
            当前模型主要基于英文招聘文本训练，中文可展示流程，但英文输入更接近训练分布。
          </p>
          <div class="signal-strip">
            <div class="signal-unit">
              <div class="signal-label">Model</div>
              <div class="signal-value">CountVectorizer + LogisticRegression</div>
            </div>
            <div class="signal-unit">
              <div class="signal-label">Notebook Baseline</div>
              <div class="signal-value">5-fold AUC 0.851820</div>
            </div>
            <div class="signal-unit">
              <div class="signal-label">Dataset Context</div>
              <div class="signal-value">~18K postings, highly imbalanced target</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel() -> bool:
    st.markdown('<div class="panel-title">Job Post Input / 招聘信息输入</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Use the sample buttons for a fast demo, or paste one complete job posting into the combined text box below.</div>',
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
    st.caption("This is the recommended testing mode for this baseline: paste the full posting text in one block.")
    st.caption(
        "Research note: on the original CSV, description-only recall is about 0.52 for fake jobs, while full combined text recall is about 0.94."
    )

    primary_run, secondary_run = st.columns([1, 1.1], gap="small")
    analyze_clicked = primary_run.button(
        "Analyze Post / 运行判断",
        type="primary",
        use_container_width=True,
    )
    secondary_run.markdown(
        '<p class="subtle" style="padding-top:0.55rem;">Paste the full job post text for the closest match to training-time behavior.</p>',
        unsafe_allow_html=True,
    )
    return auto_predict or analyze_clicked


def render_result_panel(bundle, bundle_error: str, should_predict: bool) -> None:
    st.markdown('<div class="panel-title">Prediction Desk / 结果面板</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Read the score, then inspect which tokens most strongly support fraud vs legitimacy.</div>',
        unsafe_allow_html=True,
    )

    if should_predict and not bundle_error:
        combined_text = st.session_state.get("combined_text", "").strip()
        if combined_text:
            try:
                st.session_state.last_result = predict_raw_text(combined_text, bundle)
                st.session_state.last_error = ""
            except Exception as exc:
                st.session_state.last_result = None
                st.session_state.last_error = str(exc)
        else:
            st.session_state.last_result = None
            st.session_state.last_error = "Please enter combined job text. / 请填写整段招聘文本。"

    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    elif st.session_state.last_result is None:
        st.markdown(
            """
            <div class="info-note">
              <strong>How to use / 使用方式</strong><br/>
              Load one of the two curated samples or paste your own job posting. The app will score the post
              with the saved MVP artifact under <code>model/mvp</code> and surface the most influential matched terms.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        result = st.session_state.last_result
        risk_color = "#b42318" if result.label == 1 else "#166534"
        probability = result.fraud_probability * 100
        st.markdown(
            f"""
            <div class="result-label">Fraud Probability / 诈骗概率</div>
            <div class="result-risk" style="color:{risk_color};">{result.risk_label}</div>
            <div class="result-prob">{probability:.1f}% chance of fraud according to the baseline classifier.</div>
            <div class="prob-track"><div class="prob-fill" style="width:{probability:.1f}%;"></div></div>
            <div class="meta-grid">
              <div class="meta-unit">
                <div class="meta-name">Confidence</div>
                <div class="meta-value">{result.confidence_band}</div>
              </div>
              <div class="meta-unit">
                <div class="meta-name">Input Mode</div>
                <div class="meta-value">combined_text</div>
              </div>
              <div class="meta-unit">
                <div class="meta-name">Processed Tokens</div>
                <div class="meta-value">{len(result.processed_text.split())}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="term-block">
              <div class="term-title">Input Coverage / 输入覆盖字段</div>
              <div>{format_active_fields(result.active_fields)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        positive_col, negative_col = st.columns(2, gap="large")
        with positive_col:
            render_term_rows(
                "Fraud signals / 推高可疑概率的词",
                result.top_positive_terms,
                empty_text="No strong fraud-leaning terms matched this input.",
            )
        with negative_col:
            render_term_rows(
                "Legit signals / 拉低可疑概率的词",
                result.top_negative_terms,
                empty_text="No strong legitimacy-leaning terms matched this input.",
            )

        with st.expander("Processed text preview / 预处理结果预览"):
            st.code(result.processed_text[:4000] or "(empty)", language="text")


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
              <div class="term-name">{term.term}</div>
              <div class="term-score">{term.contribution:+.3f} · x{term.count}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_model_notes(bundle) -> None:
    st.markdown("---")
    note_a, note_b, note_c = st.columns([1.05, 1, 1], gap="large")

    with note_a:
        st.subheader("Model Notes / 模型说明")
        st.write(
            "This app uses the saved Kaggle MVP artifact directly. It is a lightweight inference demo, "
            "not a retraining pipeline or a full production fraud system."
        )
        st.write(
            "The baseline was trained on structured text fields merged from job title, company profile, "
            "description, requirements, and benefits."
        )
        st.write(
            "Observed on the original CSV with this exact artifact: description-only input recalls about 51.7% of fake posts, "
            "while full combined text recalls about 93.6%."
        )

    with note_b:
        st.subheader("Preprocess / 预处理")
        if bundle:
            steps = bundle.metadata.get("preprocess", [])
            st.write(" -> ".join(steps))
        else:
            st.write("Artifacts not loaded yet.")

    with note_c:
        st.subheader("Artifacts / 模型产物")
        st.write(f"Path: `{Path(DEFAULT_ARTIFACT_DIR).as_posix()}`")
        st.write("Files: `count_vectorizer.joblib`, `logreg_model.joblib`, `metadata.json`")
        st.write("BERT and GloVe stay in the notebook as experiment context only.")


def render_missing_artifacts(error_message: str) -> None:
    st.warning("Saved model artifacts are missing or unreadable.")
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
