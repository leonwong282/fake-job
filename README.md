# Fake Job Demo

Local Streamlit demo for fake-job fraud scoring with a gated model path:

- Baseline lexical artifacts live in `model/mvp`
- Multilingual primary artifacts are expected under `model/multilingual_primary`
- The web app only activates the multilingual primary model after Kaggle exports that bundle

## Run

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## Current Routing

- Baseline always available when these files exist:
  - `model/mvp/count_vectorizer.joblib`
  - `model/mvp/logreg_model.joblib`
  - `model/mvp/metadata.json`
- Multilingual primary becomes active only when these Kaggle-exported files exist:
  - `model/multilingual_primary/classifier.joblib`
  - `model/multilingual_primary/metadata.json`
  - `model/multilingual_primary/cv_results.csv`
  - `model/multilingual_primary/README.txt`
- The app keeps baseline lexical cues visible for comparison and fallback.

## Demo Scope

- Input mode: paste one full posting into the combined text box
- Active output: fraud probability and label from the current active model
- Comparison output: multilingual primary vs lexical baseline when both are available
- Explanation output: matched lexical term contributions from the baseline only

## Kaggle Export Runbook

1. Open `fake-job-export.ipynb` in Kaggle.
2. Run the multilingual export section at the end of the notebook.
3. That section compares only:
   - `bert-base-multilingual-cased`
   - `xlm-roberta-base`
4. Model selection is fixed:
   - highest mean fraud-class F1
   - tie-break by mean ROC-AUC
   - final tie-break prefers `bert-base-multilingual-cased`
5. The notebook exports the winning classifier head and metadata bundle to `model/multilingual_primary/`.
6. Download that directory from Kaggle and place it into this repo at `model/multilingual_primary/`.

The exported `metadata.json` is expected to contain:

- `hf_model_name`
- `model_type`
- `text_fields`
- `max_len`
- `threshold`
- `selection_metric`
- `training_source`
- `preprocess_mode`

The repo does not store full Hugging Face weights. The web app reads `hf_model_name` from `metadata.json`, downloads the backbone on first use, and then relies on the local Hugging Face cache.

## Local Runtime And Fallback

- If `model/multilingual_primary/` is missing or unreadable, the app keeps using the baseline.
- If the multilingual classifier bundle exists but runtime loading fails, the app falls back to the baseline and shows the reason in the UI.
- If both multilingual and baseline are available, the active prediction uses the multilingual primary model and the UI still renders baseline lexical cues as a transparency aid.
- If baseline artifacts are missing, the app shows a baseline artifact warning instead of lexical explanations.

## Notes

- The multilingual primary model is a transformer feature extractor plus a `LogisticRegression` classifier head, not a full fine-tuned transformer.
- Baseline preprocessing still follows the saved MVP artifact metadata.
- On first multilingual inference run, model download time depends on your network and Hugging Face cache state.
