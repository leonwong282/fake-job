# Fake Job Demo

Local Streamlit demo for fake-job fraud scoring with a fixed four-model compare flow:

- `model/count_lr`
- `model/tfidf_lr`
- `model/distilbert_lr`
- `model/multilingual_primary`

The web app runs all four models on the same posting, keeps `tfidf_lr` selected by default for the main verdict card, and shows a compare surface for `count_lr`, `tfidf_lr`, `distilbert_lr`, and `multilingual_primary` on every analysis.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

If you want to pin transformer inference to a specific interpreter:

```bash
FAKE_JOB_TRANSFORMER_PYTHON=/usr/local/bin/python3 python3 -m streamlit run app.py
```

## Demo Contract

- Input modes:
  - combined full-text paste
  - structured field entry for `title`, `company_profile`, `description`, `requirements`, `benefits`
- Model selector:
  - the selected model owns the primary verdict card
  - the compare surface still renders every model run
- Language routing:
  - English and Latin-script input runs all four models
  - non-Latin input keeps `multilingual_primary` active
  - `count_lr`, `tfidf_lr`, and `distilbert_lr` are shown as skipped on non-Latin input instead of returning misleading scores
  - low-score non-Latin results from `multilingual_primary` are shown as `Needs Review`, not `Likely Legit`

## Supported Models

### `count_lr`

Expected files:

- `model/count_lr/vectorizer.joblib`
- `model/count_lr/classifier.joblib`
- `model/count_lr/metadata.json`

Behavior:

- lexical model
- English-oriented
- shows lexical term contributions in the selected-model detail area

### `tfidf_lr`

Expected files:

- `model/tfidf_lr/vectorizer.joblib`
- `model/tfidf_lr/classifier.joblib`
- `model/tfidf_lr/metadata.json`

Behavior:

- lexical model
- English-oriented
- default selected model in the UI
- shows lexical term contributions in the selected-model detail area

### `distilbert_lr`

Expected files:

- `model/distilbert_lr/classifier.joblib`
- `model/distilbert_lr/metadata.json`

Metadata requirements:

- `encoder_type=transformer_embedding`
- `model_name`
- `max_len`
- `threshold`

Behavior:

- transformer embedding model
- English-oriented
- does not show lexical term explanations
- the selected-model detail area shows model input preview instead
- may appear as `Unavailable` on machines where local transformer inference fails

### `multilingual_primary`

Expected files:

- `model/multilingual_primary/classifier.joblib`
- `model/multilingual_primary/metadata.json`
- `model/multilingual_primary/cv_results.csv`
- `model/multilingual_primary/README.txt`

Metadata requirements:

- `hf_model_name`
- `model_type`
- `text_fields`
- `max_len`
- `threshold`
- `selection_metric`
- `training_source`
- `preprocess_mode=raw_multilingual_text`

Behavior:

- multilingual transformer CLS embedding model
- accepts multilingual input
- uses the raw joined text fields instead of the English lexical preprocessing path
- treated as an experimental cross-lingual score in the UI
- does not show lexical term explanations
- the selected-model detail area shows model input preview instead
- for non-Latin input, scores below threshold are surfaced as `Needs Review` instead of `Likely Legit`
- for non-Latin input, the verdict card shows `Confidence: Experimental`
- may appear as `Unavailable` on machines where the local mBERT forward pass crashes

## Runtime Notes

- `glove_nn` is intentionally out of scope for the web demo.
- The repo stores classifier heads and metadata, not full Hugging Face weights.
- Transformer models download or read their backbones from the local Hugging Face cache on first use.
- DistilBERT and the multilingual model run in isolated subprocesses so lexical models can still return results if a transformer path fails at runtime.

## Explanation Rules

- If the selected model is `count_lr`, the UI shows lexical term contributions from the CountVectorizer features.
- If the selected model is `tfidf_lr`, the UI shows lexical term contributions from TF-IDF feature weights.
- If the selected model is `distilbert_lr` or `multilingual_primary`, the UI avoids fabricated lexical explanations and shows model input preview instead.
- The current version does not interpret a low multilingual score on Chinese or other non-Latin input as proof that a posting is legitimate.
