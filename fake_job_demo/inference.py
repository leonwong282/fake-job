from __future__ import annotations

import json
import os
import re
import shutil
import string
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

from .stopwords_en import NLTK_ENGLISH_STOPWORDS

MODEL_TEXT_FIELDS = (
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
)

FIELD_LABELS = {
    "title": "Title",
    "company_profile": "Company Profile",
    "description": "Description",
    "requirements": "Requirements",
    "benefits": "Benefits",
}

COUNT_LR_MODEL_ID = "count_lr"
TFIDF_LR_MODEL_ID = "tfidf_lr"
DISTILBERT_LR_MODEL_ID = "distilbert_lr"
MULTILINGUAL_PRIMARY_MODEL_ID = "multilingual_primary"
DEFAULT_SELECTED_MODEL_ID = TFIDF_LR_MODEL_ID

BASELINE_MODEL_ID = COUNT_LR_MODEL_ID
PRIMARY_MODEL_ID = DISTILBERT_LR_MODEL_ID

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "model"
COUNT_LR_ARTIFACT_DIR = MODEL_ROOT / COUNT_LR_MODEL_ID
TFIDF_LR_ARTIFACT_DIR = MODEL_ROOT / TFIDF_LR_MODEL_ID
DISTILBERT_LR_ARTIFACT_DIR = MODEL_ROOT / DISTILBERT_LR_MODEL_ID
MULTILINGUAL_PRIMARY_ARTIFACT_DIR = MODEL_ROOT / MULTILINGUAL_PRIMARY_MODEL_ID

DEFAULT_ARTIFACT_DIR = MODEL_ROOT / "mvp"
DEFAULT_MULTILINGUAL_ARTIFACT_DIR = MODEL_ROOT / "multilingual_primary"

MODEL_FAMILY_LEXICAL = "lexical"
MODEL_FAMILY_TRANSFORMER = "transformer_embedding"
MODEL_FAMILY_MULTILINGUAL = "multilingual_transformer"

TRANSFORMER_INPUT_PROCESSED_ENGLISH = "processed_english_text"
TRANSFORMER_INPUT_RAW_MULTILINGUAL = "raw_multilingual_text"

SUPPORTED_MULTILINGUAL_MODEL_NAMES = frozenset(
    {
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
    }
)


@dataclass(frozen=True)
class TermContribution:
    term: str
    weight: float
    feature_value: float
    feature_kind: str
    contribution: float
    direction: str


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    model_label: str
    model_type: str
    label: int
    risk_label: str
    fraud_probability: float
    threshold: float
    confidence_band: str
    processed_text: str
    raw_text: str
    model_input_text: str
    active_fields: tuple[str, ...]
    top_positive_terms: tuple[TermContribution, ...]
    top_negative_terms: tuple[TermContribution, ...]
    explanation_source: str


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_dir: Path
    vectorizer: Any
    model: Any
    metadata: dict[str, Any]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class TransformerEmbeddingBundle:
    artifact_dir: Path
    classifier: Any
    metadata: dict[str, Any]
    model_name: str
    max_len: int
    threshold: float
    input_mode: str


@dataclass(frozen=True)
class DemoModelSpec:
    model_id: str
    display_label: str
    family: str
    artifact_dir: Path
    default_selectable: bool = False


@dataclass(frozen=True)
class DemoModelState:
    spec: DemoModelSpec
    bundle: ArtifactBundle | TransformerEmbeddingBundle | None
    error_message: str

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None


@dataclass(frozen=True)
class ModelRunResult:
    model_id: str
    display_label: str
    family: str
    model_type: str
    status: str
    prediction: PredictionResult | None
    error_message: str = ""


@dataclass(frozen=True)
class TransformerBackbone:
    model_name: str
    tokenizer: Any
    model: Any
    device: str


DEMO_MODEL_SPECS: tuple[DemoModelSpec, ...] = (
    DemoModelSpec(
        model_id=COUNT_LR_MODEL_ID,
        display_label="Count LR",
        family=MODEL_FAMILY_LEXICAL,
        artifact_dir=COUNT_LR_ARTIFACT_DIR,
    ),
    DemoModelSpec(
        model_id=TFIDF_LR_MODEL_ID,
        display_label="TF-IDF LR",
        family=MODEL_FAMILY_LEXICAL,
        artifact_dir=TFIDF_LR_ARTIFACT_DIR,
        default_selectable=True,
    ),
    DemoModelSpec(
        model_id=DISTILBERT_LR_MODEL_ID,
        display_label="DistilBERT LR",
        family=MODEL_FAMILY_TRANSFORMER,
        artifact_dir=DISTILBERT_LR_ARTIFACT_DIR,
    ),
    DemoModelSpec(
        model_id=MULTILINGUAL_PRIMARY_MODEL_ID,
        display_label="Multilingual Primary",
        family=MODEL_FAMILY_MULTILINGUAL,
        artifact_dir=MULTILINGUAL_PRIMARY_ARTIFACT_DIR,
    ),
)

DEMO_MODEL_SPEC_BY_ID = {spec.model_id: spec for spec in DEMO_MODEL_SPECS}


def build_raw_text_from_fields(**job_post: str) -> str:
    parts = [str(job_post.get(field, "")).strip() for field in MODEL_TEXT_FIELDS]
    return ",".join(part for part in parts if part)


def build_multilingual_text_from_fields(**job_post: str) -> str:
    parts = [str(job_post.get(field, "")).strip() for field in MODEL_TEXT_FIELDS]
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


def preprocess_text(raw_text: str) -> str:
    text = str(raw_text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    tokens = re.findall(r"\w+", text)
    filtered = [token for token in tokens if token not in NLTK_ENGLISH_STOPWORDS]
    return " ".join(filtered)


def list_demo_models() -> tuple[DemoModelSpec, ...]:
    return DEMO_MODEL_SPECS


def get_demo_model_spec(model_id: str) -> DemoModelSpec:
    try:
        return DEMO_MODEL_SPEC_BY_ID[model_id]
    except KeyError as exc:
        known = ", ".join(DEMO_MODEL_SPEC_BY_ID)
        raise KeyError(f"Unknown model id: {model_id}. Expected one of: {known}") from exc


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _pick_existing_path(artifact_path: Path, candidates: Sequence[str], label: str) -> Path:
    for candidate in candidates:
        path = artifact_path / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {label} in {artifact_path}. Expected one of: {', '.join(candidates)}"
    )


@lru_cache(maxsize=8)
def load_artifacts(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> ArtifactBundle:
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {artifact_path}")

    vectorizer_path = _pick_existing_path(
        artifact_path,
        ("vectorizer.joblib", "count_vectorizer.joblib"),
        "vectorizer artifact",
    )
    classifier_path = _pick_existing_path(
        artifact_path,
        ("classifier.joblib", "logreg_model.joblib"),
        "classifier artifact",
    )

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(classifier_path)
    metadata = _read_metadata(metadata_path)
    feature_names = tuple(vectorizer.get_feature_names_out())

    return ArtifactBundle(
        artifact_dir=artifact_path,
        vectorizer=vectorizer,
        model=model,
        metadata=metadata,
        feature_names=feature_names,
    )


@lru_cache(maxsize=4)
def load_transformer_embedding_artifacts(
    artifact_dir: str | Path = DISTILBERT_LR_ARTIFACT_DIR,
) -> TransformerEmbeddingBundle:
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "metadata.json"
    classifier_path = artifact_path / "classifier.joblib"

    missing = [
        str(path.name)
        for path in (metadata_path, classifier_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing transformer embedding artifacts in {artifact_path}: {', '.join(missing)}"
        )

    metadata = _read_metadata(metadata_path)
    required_keys = {"encoder_type", "model_name", "max_len", "threshold", "model_type"}
    missing_keys = sorted(required_keys - metadata.keys())
    if missing_keys:
        raise ValueError(
            "Transformer embedding metadata is missing required keys: "
            f"{', '.join(missing_keys)}"
        )
    if metadata["encoder_type"] != MODEL_FAMILY_TRANSFORMER:
        raise ValueError(
            "Unsupported encoder_type for transformer embedding bundle: "
            f"{metadata['encoder_type']}"
        )

    classifier = joblib.load(classifier_path)
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Loaded transformer classifier does not implement predict_proba.")

    return TransformerEmbeddingBundle(
        artifact_dir=artifact_path,
        classifier=classifier,
        metadata=metadata,
        model_name=str(metadata["model_name"]),
        max_len=int(metadata["max_len"]),
        threshold=float(metadata["threshold"]),
        input_mode=TRANSFORMER_INPUT_PROCESSED_ENGLISH,
    )


@lru_cache(maxsize=4)
def load_multilingual_transformer_artifacts(
    artifact_dir: str | Path = DEFAULT_MULTILINGUAL_ARTIFACT_DIR,
) -> TransformerEmbeddingBundle:
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "metadata.json"
    classifier_path = artifact_path / "classifier.joblib"

    missing = [
        str(path.name)
        for path in (metadata_path, classifier_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing multilingual artifacts in {artifact_path}: {', '.join(missing)}"
        )

    metadata = _read_metadata(metadata_path)
    required_keys = {
        "hf_model_name",
        "model_type",
        "text_fields",
        "max_len",
        "threshold",
        "selection_metric",
        "training_source",
        "preprocess_mode",
    }
    missing_keys = sorted(required_keys - metadata.keys())
    if missing_keys:
        raise ValueError(
            "Multilingual metadata is missing required keys: "
            f"{', '.join(missing_keys)}"
        )

    model_name = str(metadata["hf_model_name"]).strip()
    if model_name not in SUPPORTED_MULTILINGUAL_MODEL_NAMES:
        expected = ", ".join(sorted(SUPPORTED_MULTILINGUAL_MODEL_NAMES))
        raise ValueError(
            f"Unsupported multilingual backbone {model_name}. Expected one of: {expected}"
        )

    if str(metadata["preprocess_mode"]).strip() != TRANSFORMER_INPUT_RAW_MULTILINGUAL:
        raise ValueError(
            "Unsupported multilingual preprocess_mode: "
            f"{metadata['preprocess_mode']}"
        )

    classifier = joblib.load(classifier_path)
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Loaded multilingual classifier does not implement predict_proba.")

    return TransformerEmbeddingBundle(
        artifact_dir=artifact_path,
        classifier=classifier,
        metadata=metadata,
        model_name=model_name,
        max_len=int(metadata["max_len"]),
        threshold=float(metadata["threshold"]),
        input_mode=TRANSFORMER_INPUT_RAW_MULTILINGUAL,
    )


@lru_cache(maxsize=8)
def load_demo_model_state(model_id: str) -> DemoModelState:
    spec = get_demo_model_spec(model_id)
    try:
        if spec.family == MODEL_FAMILY_LEXICAL:
            bundle = load_artifacts(spec.artifact_dir)
        elif spec.family == MODEL_FAMILY_TRANSFORMER:
            bundle = load_transformer_embedding_artifacts(spec.artifact_dir)
        elif spec.family == MODEL_FAMILY_MULTILINGUAL:
            bundle = load_multilingual_transformer_artifacts(spec.artifact_dir)
        else:
            raise ValueError(f"Unsupported model family: {spec.family}")
        return DemoModelState(spec=spec, bundle=bundle, error_message="")
    except Exception as exc:
        return DemoModelState(spec=spec, bundle=None, error_message=str(exc))


def load_demo_model_states() -> tuple[DemoModelState, ...]:
    return tuple(load_demo_model_state(spec.model_id) for spec in DEMO_MODEL_SPECS)


@lru_cache(maxsize=4)
def load_transformer_backbone(model_name: str) -> TransformerBackbone:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer inference dependencies are not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = None
    model = None
    load_errors: list[str] = []

    for local_files_only in (False, True):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            model = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            ).to(device)
            model.eval()
            break
        except Exception as exc:
            mode = "local cache only" if local_files_only else "network or cache"
            load_errors.append(f"{mode}: {exc}")

    if tokenizer is None or model is None:
        raise RuntimeError(
            f"Unable to load transformer backbone {model_name}. " + " | ".join(load_errors)
        )

    return TransformerBackbone(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def extract_transformer_features(
    texts: Sequence[str],
    model_name: str,
    max_len: int,
) -> Any:
    import torch

    backbone = load_transformer_backbone(model_name)
    if backbone.device == "mps":
        torch.mps.empty_cache()

    encoded = backbone.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    encoded = {key: value.to(backbone.device) for key, value in encoded.items()}

    with torch.inference_mode():
        outputs = backbone.model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

    features = cls_embeddings.detach().cpu().numpy()

    if backbone.device == "mps":
        torch.mps.empty_cache()

    return features


def _feature_kind(bundle: ArtifactBundle) -> str:
    vectorizer_name = bundle.vectorizer.__class__.__name__.lower()
    if "tfidf" in vectorizer_name:
        return "tfidf"
    return "count"


def explain_prediction(
    processed_text: str, bundle: ArtifactBundle, top_k: int = 8
) -> tuple[tuple[TermContribution, ...], tuple[TermContribution, ...]]:
    if not processed_text.strip():
        return (), ()

    encoded = bundle.vectorizer.transform([processed_text]).tocoo()
    coefficients = bundle.model.coef_[0]
    feature_kind = _feature_kind(bundle)
    items: list[TermContribution] = []

    for feature_index, feature_value in zip(encoded.col, encoded.data):
        weight = float(coefficients[feature_index])
        numeric_value = float(feature_value)
        contribution = weight * numeric_value
        if contribution == 0:
            continue
        items.append(
            TermContribution(
                term=bundle.feature_names[feature_index],
                weight=weight,
                feature_value=numeric_value,
                feature_kind=feature_kind,
                contribution=contribution,
                direction="fraud" if contribution > 0 else "legit",
            )
        )

    positive = tuple(
        sorted(
            (item for item in items if item.contribution > 0),
            key=lambda item: item.contribution,
            reverse=True,
        )[:top_k]
    )
    negative = tuple(
        sorted(
            (item for item in items if item.contribution < 0),
            key=lambda item: item.contribution,
        )[:top_k]
    )
    return positive, negative


def _label_from_bundle(bundle: ArtifactBundle, fallback: str) -> str:
    return str(bundle.metadata.get("model_type", fallback))


def _normalize_raw_transformer_text(raw_text: str) -> str:
    return re.sub(r"\s+", " ", str(raw_text)).strip()


def _prepare_transformer_input(
    raw_text: str,
    input_mode: str = TRANSFORMER_INPUT_PROCESSED_ENGLISH,
) -> tuple[str, str]:
    if input_mode == TRANSFORMER_INPUT_RAW_MULTILINGUAL:
        model_input_text = _normalize_raw_transformer_text(raw_text)
        return "", model_input_text

    processed_text = preprocess_text(raw_text)
    model_input_text = processed_text or _normalize_raw_transformer_text(raw_text)
    return processed_text, model_input_text


def predict_raw_text(
    raw_text: str,
    bundle: ArtifactBundle | None = None,
) -> PredictionResult:
    artifact_bundle = bundle or load_artifacts()
    processed_text = preprocess_text(raw_text)
    encoded = artifact_bundle.vectorizer.transform([processed_text])
    probability = float(artifact_bundle.model.predict_proba(encoded)[0, 1])
    threshold = float(artifact_bundle.metadata.get("threshold", 0.5))
    label = int(probability >= threshold)
    positive, negative = explain_prediction(processed_text, artifact_bundle)
    model_label = _label_from_bundle(artifact_bundle, "Lexical LogisticRegression")

    return PredictionResult(
        model_id=str(artifact_bundle.metadata.get("model_key", artifact_bundle.artifact_dir.name)),
        model_label=model_label,
        model_type=model_label,
        label=label,
        risk_label=_risk_label(label),
        fraud_probability=probability,
        threshold=threshold,
        confidence_band=_confidence_band(probability),
        processed_text=processed_text,
        raw_text=str(raw_text),
        model_input_text=processed_text,
        active_fields=("combined_text",),
        top_positive_terms=positive,
        top_negative_terms=negative,
        explanation_source="lexical_terms",
    )


def predict_job_post(
    job_post: Mapping[str, str],
    bundle: ArtifactBundle | None = None,
) -> PredictionResult:
    raw_text = build_raw_text_from_fields(**job_post)
    result = predict_raw_text(raw_text, bundle)
    active_fields = tuple(
        field for field in MODEL_TEXT_FIELDS if str(job_post.get(field, "")).strip()
    )
    return PredictionResult(
        model_id=result.model_id,
        model_label=result.model_label,
        model_type=result.model_type,
        label=result.label,
        risk_label=result.risk_label,
        fraud_probability=result.fraud_probability,
        threshold=result.threshold,
        confidence_band=result.confidence_band,
        processed_text=result.processed_text,
        raw_text=result.raw_text,
        model_input_text=result.model_input_text,
        active_fields=active_fields,
        top_positive_terms=result.top_positive_terms,
        top_negative_terms=result.top_negative_terms,
        explanation_source=result.explanation_source,
    )


def predict_transformer_raw_text(
    raw_text: str,
    bundle: TransformerEmbeddingBundle | None = None,
    use_subprocess: bool = True,
) -> PredictionResult:
    if use_subprocess:
        return _predict_transformer_raw_text_subprocess(raw_text, bundle)
    return _predict_transformer_raw_text_local(raw_text, bundle)


def _predict_transformer_raw_text_local(
    raw_text: str,
    bundle: TransformerEmbeddingBundle | None = None,
) -> PredictionResult:
    artifact_bundle = bundle or load_transformer_embedding_artifacts()
    processed_text, model_input_text = _prepare_transformer_input(
        raw_text,
        artifact_bundle.input_mode,
    )
    features = extract_transformer_features(
        [model_input_text],
        model_name=artifact_bundle.model_name,
        max_len=artifact_bundle.max_len,
    )
    probability = float(artifact_bundle.classifier.predict_proba(features)[0, 1])
    label = int(probability >= artifact_bundle.threshold)

    return PredictionResult(
        model_id=str(
            artifact_bundle.metadata.get("model_key", artifact_bundle.artifact_dir.name)
        ),
        model_label=str(artifact_bundle.metadata.get("model_type", "Transformer Embedding")),
        model_type=str(artifact_bundle.metadata.get("model_type", "Transformer Embedding")),
        label=label,
        risk_label=_risk_label(label),
        fraud_probability=probability,
        threshold=artifact_bundle.threshold,
        confidence_band=_confidence_band(probability),
        processed_text=processed_text,
        raw_text=str(raw_text),
        model_input_text=model_input_text,
        active_fields=("combined_text",),
        top_positive_terms=(),
        top_negative_terms=(),
        explanation_source="not_available",
    )


def _predict_transformer_raw_text_subprocess(
    raw_text: str,
    bundle: TransformerEmbeddingBundle | None = None,
) -> PredictionResult:
    artifact_bundle = bundle or load_transformer_embedding_artifacts()
    processed_text, model_input_text = _prepare_transformer_input(
        raw_text,
        artifact_bundle.input_mode,
    )
    python_executable = resolve_transformer_python_executable()
    command = [
        python_executable,
        "-c",
        (
            "from fake_job_demo.inference import _run_transformer_embedding_cli; "
            "import sys; "
            "raise SystemExit(_run_transformer_embedding_cli(sys.argv[1]))"
        ),
        str(artifact_bundle.artifact_dir),
    ]
    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    completed = subprocess.run(
        command,
        input=model_input_text,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=180,
        check=False,
    )

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        details = details[-800:] if details else f"exit code {completed.returncode}"
        raise RuntimeError(
            "Transformer subprocess failed. "
            f"Python: {python_executable}. "
            f"Return code: {completed.returncode}. Details: {details}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Transformer subprocess returned invalid JSON output: "
            f"{completed.stdout[-400:]}"
        ) from exc

    probability = float(payload["probability"])
    label = int(probability >= artifact_bundle.threshold)
    return PredictionResult(
        model_id=str(
            artifact_bundle.metadata.get("model_key", artifact_bundle.artifact_dir.name)
        ),
        model_label=str(artifact_bundle.metadata.get("model_type", "Transformer Embedding")),
        model_type=str(artifact_bundle.metadata.get("model_type", "Transformer Embedding")),
        label=label,
        risk_label=_risk_label(label),
        fraud_probability=probability,
        threshold=artifact_bundle.threshold,
        confidence_band=_confidence_band(probability),
        processed_text=processed_text,
        raw_text=str(raw_text),
        model_input_text=model_input_text,
        active_fields=("combined_text",),
        top_positive_terms=(),
        top_negative_terms=(),
        explanation_source="not_available",
    )


def predict_transformer_job_post(
    job_post: Mapping[str, str],
    bundle: TransformerEmbeddingBundle | None = None,
    use_subprocess: bool = True,
) -> PredictionResult:
    artifact_bundle = bundle or load_transformer_embedding_artifacts()
    if artifact_bundle.input_mode == TRANSFORMER_INPUT_RAW_MULTILINGUAL:
        raw_text = build_multilingual_text_from_fields(**job_post)
    else:
        raw_text = build_raw_text_from_fields(**job_post)
    result = predict_transformer_raw_text(
        raw_text,
        artifact_bundle,
        use_subprocess=use_subprocess,
    )
    active_fields = tuple(
        field for field in MODEL_TEXT_FIELDS if str(job_post.get(field, "")).strip()
    )
    return PredictionResult(
        model_id=result.model_id,
        model_label=result.model_label,
        model_type=result.model_type,
        label=result.label,
        risk_label=result.risk_label,
        fraud_probability=result.fraud_probability,
        threshold=result.threshold,
        confidence_band=result.confidence_band,
        processed_text=result.processed_text,
        raw_text=result.raw_text,
        model_input_text=result.model_input_text,
        active_fields=active_fields,
        top_positive_terms=result.top_positive_terms,
        top_negative_terms=result.top_negative_terms,
        explanation_source=result.explanation_source,
    )


def _predict_with_state(
    state: DemoModelState,
    *,
    raw_text: str | None = None,
    job_post: Mapping[str, str] | None = None,
) -> ModelRunResult:
    model_type = _state_model_type(state)
    input_text = _build_runtime_input_preview(raw_text, job_post)

    if state.bundle is None:
        return ModelRunResult(
            model_id=state.spec.model_id,
            display_label=state.spec.display_label,
            family=state.spec.family,
            model_type=model_type,
            status="unavailable",
            prediction=None,
            error_message=state.error_message or "Artifacts are unavailable.",
        )

    if _contains_non_latin_script(input_text) and not _supports_non_latin_input(state.spec):
        return ModelRunResult(
            model_id=state.spec.model_id,
            display_label=state.spec.display_label,
            family=state.spec.family,
            model_type=model_type,
            status="skipped",
            prediction=None,
            error_message="Skipped for non-Latin input. This model is English-only.",
        )

    try:
        if state.spec.family == MODEL_FAMILY_LEXICAL:
            bundle = state.bundle
            assert isinstance(bundle, ArtifactBundle)
            prediction = (
                predict_job_post(job_post, bundle)
                if job_post is not None
                else predict_raw_text(raw_text or "", bundle)
            )
        elif state.spec.family in {MODEL_FAMILY_TRANSFORMER, MODEL_FAMILY_MULTILINGUAL}:
            bundle = state.bundle
            assert isinstance(bundle, TransformerEmbeddingBundle)
            prediction = (
                predict_transformer_job_post(job_post, bundle)
                if job_post is not None
                else predict_transformer_raw_text(raw_text or "", bundle)
            )
        else:
            raise ValueError(f"Unsupported model family: {state.spec.family}")
        return ModelRunResult(
            model_id=state.spec.model_id,
            display_label=state.spec.display_label,
            family=state.spec.family,
            model_type=prediction.model_type,
            status="ready",
            prediction=prediction,
        )
    except Exception as exc:
        return ModelRunResult(
            model_id=state.spec.model_id,
            display_label=state.spec.display_label,
            family=state.spec.family,
            model_type=model_type,
            status="unavailable",
            prediction=None,
            error_message=str(exc),
        )


def run_demo_models_raw_text(
    raw_text: str,
    model_states: Sequence[DemoModelState] | None = None,
) -> tuple[ModelRunResult, ...]:
    states = tuple(model_states) if model_states is not None else load_demo_model_states()
    return tuple(_predict_with_state(state, raw_text=raw_text) for state in states)


def run_demo_models_job_post(
    job_post: Mapping[str, str],
    model_states: Sequence[DemoModelState] | None = None,
) -> tuple[ModelRunResult, ...]:
    states = tuple(model_states) if model_states is not None else load_demo_model_states()
    return tuple(_predict_with_state(state, job_post=job_post) for state in states)


def _risk_label(label: int) -> str:
    return "Suspicious" if label == 1 else "Likely Legit"


def _confidence_band(probability: float) -> str:
    if probability >= 0.85 or probability <= 0.15:
        return "High"
    if probability >= 0.65 or probability <= 0.35:
        return "Moderate"
    return "Low"


def _run_transformer_embedding_cli(artifact_dir: str) -> int:
    bundle = load_runtime_transformer_bundle(artifact_dir)
    raw_text = sys.stdin.read()
    _, model_input_text = _prepare_transformer_input(raw_text, bundle.input_mode)
    if not model_input_text:
        raise ValueError("No transformer input text was provided on stdin.")

    features = extract_transformer_features(
        [model_input_text],
        model_name=bundle.model_name,
        max_len=bundle.max_len,
    )
    probability = float(bundle.classifier.predict_proba(features)[0, 1])
    print(json.dumps({"probability": probability}))
    return 0


def load_runtime_transformer_bundle(artifact_dir: str | Path) -> TransformerEmbeddingBundle:
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {artifact_path}")
    metadata = _read_metadata(metadata_path)
    if "hf_model_name" in metadata:
        return load_multilingual_transformer_artifacts(artifact_path)
    return load_transformer_embedding_artifacts(artifact_path)


@lru_cache(maxsize=1)
def resolve_transformer_python_executable() -> str:
    override = os.environ.get("FAKE_JOB_TRANSFORMER_PYTHON", "").strip()
    candidates = [
        override,
        sys.executable,
        shutil.which("python3") or "",
        shutil.which("python") or "",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
    ]

    seen: set[str] = set()
    normalized_candidates: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        resolved = str(Path(candidate).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized_candidates.append(resolved)

    probe = (
        "import json, sys; "
        "out={'executable': sys.executable}; "
        "import joblib, sklearn, torch, transformers; "
        "out['torch']=torch.__version__; "
        "out['transformers']=transformers.__version__; "
        "print(json.dumps(out))"
    )

    failures: list[str] = []
    for candidate in normalized_candidates:
        if not Path(candidate).exists():
            failures.append(f"{candidate}: missing")
            continue
        completed = subprocess.run(
            [candidate, "-c", probe],
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=20,
            check=False,
        )
        if completed.returncode == 0:
            return candidate

        details = (completed.stderr or completed.stdout or "").strip()
        details = details[-240:] if details else f"exit code {completed.returncode}"
        failures.append(f"{candidate}: {details}")

    raise RuntimeError(
        "No Python interpreter with transformer dependencies was found. "
        "Set FAKE_JOB_TRANSFORMER_PYTHON to a working interpreter. "
        "Tried: " + " | ".join(failures)
    )


def _state_model_type(state: DemoModelState) -> str:
    bundle = state.bundle
    if isinstance(bundle, ArtifactBundle):
        return str(bundle.metadata.get("model_type", state.spec.display_label))
    if isinstance(bundle, TransformerEmbeddingBundle):
        return str(bundle.metadata.get("model_type", state.spec.display_label))
    return format_model_family_name(state.spec.family)


def format_model_family_name(family: str) -> str:
    if family == MODEL_FAMILY_MULTILINGUAL:
        return "Multilingual transformer"
    if family == MODEL_FAMILY_TRANSFORMER:
        return "Transformer embedding"
    return "Lexical"


def _build_runtime_input_preview(
    raw_text: str | None,
    job_post: Mapping[str, str] | None,
) -> str:
    if job_post is not None:
        return "\n".join(
            str(job_post.get(field, "")).strip()
            for field in MODEL_TEXT_FIELDS
            if str(job_post.get(field, "")).strip()
        )
    return str(raw_text or "")


def _contains_non_latin_script(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0400-\u052F\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]",
            text,
        )
    )


def _supports_non_latin_input(spec: DemoModelSpec) -> bool:
    return spec.family == MODEL_FAMILY_MULTILINGUAL


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--predict-transformer" and sys.argv[2] == "--artifact-dir":
        raise SystemExit(_run_transformer_embedding_cli(sys.argv[3]))
