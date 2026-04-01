# Fake Job Demo

Local Streamlit demo for the saved Kaggle MVP artifacts in [model/mvp](/Users/liang/Downloads/repository/fake-job/model/mvp).

## Run

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app reads these files directly:

- `model/mvp/count_vectorizer.joblib`
- `model/mvp/logreg_model.joblib`
- `model/mvp/metadata.json`

## Demo Scope

- Input fields: `title`, `company_profile`, `description`, `requirements`, `benefits`
- Optional input mode: paste one full posting into the combined text box
- Model: `CountVectorizer + LogisticRegression`
- Output: fraud probability, predicted label, and matched term contributions

## Notes

- The UI is bilingual, but the model was trained on English-heavy job posting data.
- GloVe and BERT experiments remain in the notebook and are not used by the web demo.
- On the original CSV with this saved artifact, `description` only is much weaker than full post text. If you test with one copied block, use the combined text box.
