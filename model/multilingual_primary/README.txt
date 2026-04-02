This directory contains the Kaggle-exported multilingual primary model bundle.

Selected Hugging Face backbone: bert-base-multilingual-cased
Selection rule: highest fraud-class F1, then ROC-AUC, then prefer bert-base-multilingual-cased.

Files:
- classifier.joblib: LogisticRegression head trained on transformer CLS embeddings
- metadata.json: runtime metadata consumed by the local Streamlit app
- cv_results.csv: candidate model comparison results
- README.txt: export summary
