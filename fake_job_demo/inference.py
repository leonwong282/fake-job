from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib

from .stopwords_en import NLTK_ENGLISH_STOPWORDS

MODEL_TEXT_FIELDS = (
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "model" / "mvp"


@dataclass(frozen=True)
class TermContribution:
    term: str
    weight: float
    count: int
    contribution: float
    direction: str


@dataclass(frozen=True)
class PredictionResult:
    label: int
    risk_label: str
    fraud_probability: float
    confidence_band: str
    processed_text: str
    raw_text: str
    active_fields: tuple[str, ...]
    top_positive_terms: tuple[TermContribution, ...]
    top_negative_terms: tuple[TermContribution, ...]


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_dir: Path
    vectorizer: Any
    model: Any
    metadata: dict[str, Any]
    feature_names: tuple[str, ...]


def build_raw_text_from_fields(**job_post: str) -> str:
    parts = [str(job_post.get(field, "")).strip() for field in MODEL_TEXT_FIELDS]
    return ",".join(part for part in parts if part)


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
    label = int(artifact_bundle.model.predict(encoded)[0])
    positive, negative = explain_prediction(processed_text, artifact_bundle)

    return PredictionResult(
        label=label,
        risk_label="Suspicious / 可疑" if label == 1 else "Likely Legit / 较像真实职位",
        fraud_probability=probability,
        confidence_band=_confidence_band(probability),
        processed_text=processed_text,
        raw_text=str(raw_text),
        active_fields=("combined_text",),
        top_positive_terms=positive,
        top_negative_terms=negative,
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
        label=result.label,
        risk_label=result.risk_label,
        fraud_probability=result.fraud_probability,
        confidence_band=result.confidence_band,
        processed_text=result.processed_text,
        raw_text=result.raw_text,
        active_fields=active_fields,
        top_positive_terms=result.top_positive_terms,
        top_negative_terms=result.top_negative_terms,
    )


def _confidence_band(probability: float) -> str:
    if probability >= 0.85 or probability <= 0.15:
        return "High / 高"
    if probability >= 0.65 or probability <= 0.35:
        return "Moderate / 中"
    return "Low / 低"
