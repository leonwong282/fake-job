from __future__ import annotations

import json
import re
import string
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

BASELINE_MODEL_ID = "baseline_lexical"
PRIMARY_MODEL_ID = "primary_multilingual"

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "model" / "mvp"
DEFAULT_MULTILINGUAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "model" / "multilingual_primary"
)


@dataclass(frozen=True)
class TermContribution:
    term: str
    weight: float
    count: int
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
class MultilingualArtifactBundle:
    artifact_dir: Path
    classifier: Any
    metadata: dict[str, Any]
    hf_model_name: str
    max_len: int
    threshold: float


@dataclass(frozen=True)
class TransformerBackbone:
    model_name: str
    tokenizer: Any
    model: Any
    device: str


@dataclass(frozen=True)
class ModelBundleState:
    baseline_bundle: ArtifactBundle | None
    baseline_error: str
    multilingual_bundle: MultilingualArtifactBundle | None
    multilingual_error: str


@dataclass(frozen=True)
class DemoPredictionResult:
    active_model_id: str
    baseline_result: PredictionResult | None
    primary_result: PredictionResult | None
    fallback_reason: str

    @property
    def active_result(self) -> PredictionResult | None:
        if self.active_model_id == PRIMARY_MODEL_ID:
            return self.primary_result
        if self.active_model_id == BASELINE_MODEL_ID:
            return self.baseline_result
        return None


def build_raw_text_from_fields(**job_post: str) -> str:
    parts = [str(job_post.get(field, "")).strip() for field in MODEL_TEXT_FIELDS]
    return ",".join(part for part in parts if part)


def build_multilingual_text_from_fields(**job_post: str) -> str:
    sections = []
    for field in MODEL_TEXT_FIELDS:
        value = str(job_post.get(field, "")).strip()
        if not value:
            continue
        sections.append(f"{FIELD_LABELS[field]}: {value}")
    return "\n".join(sections)


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


@lru_cache(maxsize=4)
def load_artifacts(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> ArtifactBundle:
    artifact_path = Path(artifact_dir)
    vectorizer_path = artifact_path / "count_vectorizer.joblib"
    model_path = artifact_path / "logreg_model.joblib"
    metadata_path = artifact_path / "metadata.json"

    missing = [
        str(path.name)
        for path in (vectorizer_path, model_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in {artifact_path}: {', '.join(missing)}"
        )

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = tuple(vectorizer.get_feature_names_out())

    return ArtifactBundle(
        artifact_dir=artifact_path,
        vectorizer=vectorizer,
        model=model,
        metadata=metadata,
        feature_names=feature_names,
    )


@lru_cache(maxsize=4)
def load_multilingual_artifacts(
    artifact_dir: str | Path = DEFAULT_MULTILINGUAL_ARTIFACT_DIR,
) -> MultilingualArtifactBundle:
    artifact_path = Path(artifact_dir)
    classifier_path = artifact_path / "classifier.joblib"
    metadata_path = artifact_path / "metadata.json"

    missing = [
        str(path.name)
        for path in (classifier_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Multilingual artifacts are unavailable in "
            f"{artifact_path}: missing {', '.join(missing)}. "
            "Generate them in Kaggle and place them under model/multilingual_primary."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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

    classifier = joblib.load(classifier_path)
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Loaded multilingual classifier does not implement predict_proba.")

    return MultilingualArtifactBundle(
        artifact_dir=artifact_path,
        classifier=classifier,
        metadata=metadata,
        hf_model_name=str(metadata["hf_model_name"]),
        max_len=int(metadata["max_len"]),
        threshold=float(metadata["threshold"]),
    )


def load_model_bundle_state(
    baseline_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    multilingual_dir: str | Path = DEFAULT_MULTILINGUAL_ARTIFACT_DIR,
) -> ModelBundleState:
    baseline_bundle: ArtifactBundle | None = None
    multilingual_bundle: MultilingualArtifactBundle | None = None
    baseline_error = ""
    multilingual_error = ""

    try:
        baseline_bundle = load_artifacts(baseline_dir)
    except Exception as exc:
        baseline_error = str(exc)

    try:
        multilingual_bundle = load_multilingual_artifacts(multilingual_dir)
    except Exception as exc:
        multilingual_error = str(exc)

    return ModelBundleState(
        baseline_bundle=baseline_bundle,
        baseline_error=baseline_error,
        multilingual_bundle=multilingual_bundle,
        multilingual_error=multilingual_error,
    )


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
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
    encoded = backbone.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    encoded = {key: value.to(backbone.device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = backbone.model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

    return cls_embeddings.cpu().numpy()


def explain_prediction(
    processed_text: str, bundle: ArtifactBundle, top_k: int = 8
) -> tuple[tuple[TermContribution, ...], tuple[TermContribution, ...]]:
    if not processed_text.strip():
        return (), ()

    encoded = bundle.vectorizer.transform([processed_text]).tocoo()
    coefficients = bundle.model.coef_[0]
    items: list[TermContribution] = []

    for feature_index, count in zip(encoded.col, encoded.data):
        weight = float(coefficients[feature_index])
        contribution = weight * int(count)
        if contribution == 0:
            continue

        items.append(
            TermContribution(
                term=bundle.feature_names[feature_index],
                weight=weight,
                count=int(count),
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


def predict_raw_text(
    raw_text: str, bundle: ArtifactBundle | None = None
) -> PredictionResult:
    artifact_bundle = bundle or load_artifacts()
    processed_text = preprocess_text(raw_text)
    encoded = artifact_bundle.vectorizer.transform([processed_text])
    probability = float(artifact_bundle.model.predict_proba(encoded)[0, 1])
    threshold = 0.5
    label = int(probability >= threshold)
    positive, negative = explain_prediction(processed_text, artifact_bundle)

    return PredictionResult(
        model_id=BASELINE_MODEL_ID,
        model_label="CountVectorizer + LogisticRegression",
        model_type=str(
            artifact_bundle.metadata.get("model_type", "CountVectorizer + LogisticRegression")
        ),
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
        explanation_source="baseline_lexical_terms",
    )


def predict_job_post(
    job_post: Mapping[str, str], bundle: ArtifactBundle | None = None
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


def predict_multilingual_raw_text(
    raw_text: str,
    bundle: MultilingualArtifactBundle | None = None,
) -> PredictionResult:
    artifact_bundle = bundle or load_multilingual_artifacts()
    model_input_text = str(raw_text).strip()
    features = extract_transformer_features(
        [model_input_text],
        model_name=artifact_bundle.hf_model_name,
        max_len=artifact_bundle.max_len,
    )
    probability = float(artifact_bundle.classifier.predict_proba(features)[0, 1])
    label = int(probability >= artifact_bundle.threshold)

    return PredictionResult(
        model_id=PRIMARY_MODEL_ID,
        model_label="Multilingual Transformer + LogisticRegression",
        model_type=str(artifact_bundle.metadata["model_type"]),
        label=label,
        risk_label=_risk_label(label),
        fraud_probability=probability,
        threshold=artifact_bundle.threshold,
        confidence_band=_confidence_band(probability),
        processed_text="",
        raw_text=str(raw_text),
        model_input_text=model_input_text,
        active_fields=("combined_text",),
        top_positive_terms=(),
        top_negative_terms=(),
        explanation_source="not_available",
    )


def predict_multilingual_job_post(
    job_post: Mapping[str, str],
    bundle: MultilingualArtifactBundle | None = None,
) -> PredictionResult:
    raw_text = build_multilingual_text_from_fields(**job_post)
    result = predict_multilingual_raw_text(raw_text, bundle)
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


def predict_demo_raw_text(
    raw_text: str,
    model_state: ModelBundleState | None = None,
) -> DemoPredictionResult:
    state = model_state or load_model_bundle_state()
    baseline_result: PredictionResult | None = None
    primary_result: PredictionResult | None = None
    fallback_reason = ""

    if state.baseline_bundle is not None:
        baseline_result = predict_raw_text(raw_text, state.baseline_bundle)

    if state.multilingual_bundle is not None:
        try:
            primary_result = predict_multilingual_raw_text(raw_text, state.multilingual_bundle)
        except Exception as exc:
            fallback_reason = f"Multilingual model unavailable at runtime: {exc}"
    elif state.multilingual_error:
        fallback_reason = state.multilingual_error

    if primary_result is not None:
        return DemoPredictionResult(
            active_model_id=PRIMARY_MODEL_ID,
            baseline_result=baseline_result,
            primary_result=primary_result,
            fallback_reason=fallback_reason,
        )

    if baseline_result is not None:
        return DemoPredictionResult(
            active_model_id=BASELINE_MODEL_ID,
            baseline_result=baseline_result,
            primary_result=None,
            fallback_reason=fallback_reason,
        )

    raise RuntimeError(
        state.baseline_error
        or fallback_reason
        or "No model artifacts are available for inference."
    )


def predict_demo_job_post(
    job_post: Mapping[str, str],
    model_state: ModelBundleState | None = None,
) -> DemoPredictionResult:
    state = model_state or load_model_bundle_state()
    baseline_result: PredictionResult | None = None
    primary_result: PredictionResult | None = None
    fallback_reason = ""

    if state.baseline_bundle is not None:
        baseline_result = predict_job_post(job_post, state.baseline_bundle)

    if state.multilingual_bundle is not None:
        try:
            primary_result = predict_multilingual_job_post(
                job_post,
                state.multilingual_bundle,
            )
        except Exception as exc:
            fallback_reason = f"Multilingual model unavailable at runtime: {exc}"
    elif state.multilingual_error:
        fallback_reason = state.multilingual_error

    if primary_result is not None:
        return DemoPredictionResult(
            active_model_id=PRIMARY_MODEL_ID,
            baseline_result=baseline_result,
            primary_result=primary_result,
            fallback_reason=fallback_reason,
        )

    if baseline_result is not None:
        return DemoPredictionResult(
            active_model_id=BASELINE_MODEL_ID,
            baseline_result=baseline_result,
            primary_result=None,
            fallback_reason=fallback_reason,
        )

    raise RuntimeError(
        state.baseline_error
        or fallback_reason
        or "No model artifacts are available for inference."
    )


def _risk_label(label: int) -> str:
    return "Suspicious / 可疑" if label == 1 else "Likely Legit / 较像真实职位"


def _confidence_band(probability: float) -> str:
    if probability >= 0.85 or probability <= 0.15:
        return "High / 高"
    if probability >= 0.65 or probability <= 0.35:
        return "Moderate / 中"
    return "Low / 低"
