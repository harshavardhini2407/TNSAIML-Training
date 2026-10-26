# Credit Card Fraud Detection Using Classification Techniques

A complete beginner-friendly ML project with a Streamlit dashboard. It trains and compares Logistic Regression and Random Forest classifiers, evaluates them using imbalanced-class metrics, saves the best preprocessing/model pipeline, and supports single-transaction and batch predictions.

## 1. Project structure

```text
credit_card_fraud_detection/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── train_model.py
└── tests/
    └── test_training_helpers.py
```

The trained model and evaluation metrics are created after you run training; they are not pre-created because they depend on your dataset.

## 2. Dataset

The project supports a CSV with a binary target column named `Class` or `Fraud` (also accepts `class`, `fraud`, `is_fraud`, or `target`):
- `0` = legitimate transaction
- `1` = fraudulent transaction

A commonly used educational dataset is the Kaggle **Credit Card Fraud Detection** dataset, which has `Time`, `V1`–`V28`, `Amount`, and `Class` columns. Obtain it from Kaggle or another source whose license permits your use. Put the CSV at `data/creditcard.csv`, or use a different filename with `--data path/to/file.csv`.

Do not commit sensitive customer or payment information to GitHub.

## 3. Setup on Windows

Open Command Prompt or the VS Code terminal in the project folder:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` does not work, try `python -m venv .venv`. Use a supported Python version compatible with the installed scikit-learn and Streamlit packages.

## 4. Train and evaluate

```bat
python src/train_model.py --data data/creditcard.csv
```

Optional settings:

```bat
python src/train_model.py --data data/creditcard.csv --test-size 0.25 --seed 7
```

The script creates:
- `models/fraud_pipeline.joblib` — fitted preprocessing and classifier pipeline plus feature schema.
- `reports/metrics.json` — evaluation results for both classifiers and the selected model.

The data is split with stratification so the rare fraud class is represented in both sets. The best model is selected by F1-score. Review precision and recall and select an alert threshold based on the intended use; there is no universally best threshold.

## 5. Run the dashboard

```bat
streamlit run app.py
```

The app opens in your browser. Sections:
- **Overview:** dataset preview, row/feature counts, missing values and class distribution.
- **Detect Fraud:** enter feature values and estimate fraud probability.
- **Batch Analysis:** upload transactions and download scored results.
- **Model Evaluation:** precision, recall, F1, ROC-AUC and confusion matrix.
- **About Project:** summary and workflow.

If no dataset is available, the app still opens; upload a CSV in the sidebar to explore it.

## 6. Important interpretation notes

Credit-card fraud datasets are typically highly imbalanced. Accuracy can look excellent even when a model misses most fraudulent transactions. Review:
- **Precision:** of flagged transactions, how many were fraud?
- **Recall:** of all fraud cases, how many did the model detect?
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC and average precision:** ranking quality, with average precision especially useful under class imbalance.
- **Confusion matrix:** false positives and false negatives.

This is an educational prototype, not a production payment-control system. It may generate false alarms or miss fraud. Do not use its output as the sole basis for blocking real transactions. Production use requires temporal validation, calibrated thresholds, monitoring, privacy controls, security review and human/operational safeguards.

## 7. Troubleshooting

- **Dataset not found:** check the file path and CSV filename.
- **No target column found:** make sure your label column is named `Class` or `Fraud` (or one of the documented alternatives).
- **Missing values:** numeric features are median-imputed inside the saved pipeline.
- **Feature mismatch in prediction:** use the same feature names and schema used for training.
- **Module not found:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **Streamlit not recognized:** run `python -m streamlit run app.py`.

## 8. Suggested academic extensions

1. Add XGBoost or LightGBM if permitted and installed.
2. Add Precision-Recall curve and ROC curve visualizations.
3. Compare class weights with random undersampling or SMOTE (apply resampling to training data only).
4. Use time-based splitting to reduce temporal leakage.
5. Add explainability with feature importance or SHAP.
6. Add unit tests and deploy with access controls.
