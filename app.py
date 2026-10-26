from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.train_model import load_dataset, TARGET_CANDIDATES, DEFAULT_DATA_PATH

st.set_page_config(page_title="CardShield | Fraud Detection", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.main {background-color:#0b1220;}
[data-testid="stSidebar"] {background-color:#101a2b;}
div[data-testid="stMetric"] {background:#121e31;border:1px solid #27364e;padding:16px;border-radius:12px;}
.hero {padding:24px 26px;border-radius:16px;background:linear-gradient(120deg,#172b4d,#101827);border:1px solid #2c4160;margin-bottom:18px;}
.small-note {color:#9fb0c8;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = ROOT / "models" / "fraud_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
DATA_PATH = ROOT / "data" / "creditcard.csv"

@st.cache_resource
def load_bundle(path_str, mtime):
    return joblib.load(path_str)

def get_bundle():
    if not MODEL_PATH.exists():
        return None
    return load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

def money(x):
    return f"${x:,.2f}"

st.markdown("""
<div class="hero">
  <h1>🛡️ CardShield</h1>
  <p>Credit Card Fraud Detection Using Classification Techniques</p>
  <div class="small-note">Machine-learning decision support • Transaction analytics • Model evaluation</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a section", ["Overview", "Detect Fraud", "Batch Analysis", "Model Evaluation", "About Project"])
    st.divider()
    st.caption("Dataset")
    uploaded_data = st.file_uploader("Optional: upload a CSV dataset", type=["csv"], key="sidebar_dataset")
    st.caption("Expected target column: Class (0 = legitimate, 1 = fraud).")

@st.cache_data
def read_csv(path, mtime):
    return pd.read_csv(path)

def active_data():
    if uploaded_data is not None:
        try:
            return pd.read_csv(uploaded_data)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return None
    if DATA_PATH.exists():
        try:
            return read_csv(str(DATA_PATH), DATA_PATH.stat().st_mtime)
        except Exception:
            return None
    return None

df = active_data()
bundle = get_bundle()

if page == "Overview":
    st.subheader("Transaction overview")
    if df is not None:
        target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Transactions", f"{len(df):,}")
        c2.metric("Features", f"{df.shape[1] - (1 if target else 0):,}")
        c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        fraud_count = int((pd.to_numeric(df[target], errors="coerce") == 1).sum()) if target else 0
        c4.metric("Fraud records", f"{fraud_count:,}" if target else "Target not found")
        left,right = st.columns([1.2,1])
        with left:
            st.markdown("#### Dataset preview")
            st.dataframe(df.head(10), use_container_width=True)
        with right:
            st.markdown("#### Class distribution")
            if target:
                counts = pd.to_numeric(df[target], errors="coerce").value_counts().reindex([0,1], fill_value=0)
                fig, ax = plt.subplots()
                ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values)
                ax.set_ylabel("Transactions")
                ax.set_title("Target label distribution")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(f"Fraud rate: {counts.get(1,0)/max(int(counts.sum()),1)*100:.4f}%")
            else:
                st.warning("No target column named Class or Fraud found.")
        with st.expander("Missing values by column"):
            st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    else:
        st.info("Upload a CSV from the sidebar or place the dataset at `data/creditcard.csv`.")
        st.markdown("#### Start here")
        st.code("python -m pip install -r requirements.txt\npython src/train_model.py --data data/creditcard.csv\nstreamlit run app.py", language="bash")
    if bundle:
        st.success(f"Trained model loaded: {bundle.get('model_name', 'classification pipeline')}")
    else:
        st.warning("No trained model found yet. Train one using the instructions in README.md.")

elif page == "Detect Fraud":
    st.subheader("Single transaction prediction")
    if not bundle:
        st.warning("Train the model first: `python src/train_model.py --data data/creditcard.csv`.")
        st.stop()
    feature_names = bundle["feature_names"]
    with st.form("single_transaction"):
        st.write("Enter values for the transaction features. For the common Kaggle dataset, these are Time, V1–V28, and Amount.")
        defaults = bundle.get("feature_defaults", {})
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                default = float(defaults.get(feature, 0.0))
                if feature.lower() == "amount":
                    values[feature] = st.number_input(feature, value=max(default,0.0), min_value=0.0, format="%.6f")
                else:
                    values[feature] = st.number_input(feature, value=default, format="%.6f")
        threshold = st.slider("Fraud alert threshold", 0.05, 0.95, float(bundle.get("threshold", 0.5)), 0.05)
        submitted = st.form_submit_button("Analyze transaction", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
        model = bundle["pipeline"]
        pred = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][list(model.classes_).index(1)]) if hasattr(model, "predict_proba") and 1 in model.classes_ else float(pred)
        flagged = probability >= threshold
        a,b,c = st.columns(3)
        a.metric("Fraud probability", f"{probability*100:.2f}%")
        b.metric("Threshold", f"{threshold*100:.0f}%")
        c.metric("Model class", "Fraud (1)" if pred == 1 else "Legitimate (0)")
        if flagged:
            st.error("⚠️ HIGH-RISK: this transaction meets the selected fraud alert threshold.")
        else:
            st.success("No fraud alert at the selected threshold.")
        st.caption("This is a model estimate, not proof that a transaction is fraudulent. Review with additional evidence.")

elif page == "Batch Analysis":
    st.subheader("Batch transaction screening")
    st.write("Upload a transaction CSV or use the dataset selected in the sidebar.")
    batch_file = st.file_uploader("Upload transactions to score", type=["csv"], key="batch_csv")
    if not bundle:
        st.warning("Train the model first to enable predictions.")
    elif batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            required = bundle["feature_names"]
            missing = [f for f in required if f not in batch.columns]
            if missing:
                st.error("Missing required feature columns: " + ", ".join(missing))
            else:
                X = batch[required].apply(pd.to_numeric, errors="coerce")
                if X.isna().any().any():
                    st.error("Some feature values are missing or non-numeric. Clean the file and upload again.")
                else:
                    probabilities = bundle["pipeline"].predict_proba(X)[:, list(bundle["pipeline"].classes_).index(1)]
                    scored = batch.copy()
                    scored["fraud_probability"] = probabilities
                    scored["predicted_fraud"] = (probabilities >= float(bundle.get("threshold",0.5))).astype(int)
                    st.metric("Flagged transactions", f"{int(scored['predicted_fraud'].sum()):,}")
                    st.dataframe(scored.sort_values("fraud_probability", ascending=False).head(100), use_container_width=True)
                    st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "fraud_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Unable to process the file: {e}")
    elif df is not None:
        st.info("Use the uploader above to score a separate CSV. The dataset shown on Overview is not automatically scored.")

elif page == "Model Evaluation":
    st.subheader("Classification model evaluation")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.caption(f"Evaluation model: {metrics.get('model_name','N/A')} | Dataset: {metrics.get('dataset','N/A')}")
        a,b,c,d = st.columns(4)
        a.metric("Precision", f"{metrics.get('precision',0):.4f}")
        b.metric("Recall", f"{metrics.get('recall',0):.4f}")
        c.metric("F1-score", f"{metrics.get('f1',0):.4f}")
        d.metric("ROC-AUC", f"{metrics.get('roc_auc',0):.4f}" if metrics.get("roc_auc") is not None else "N/A")
        st.info("For highly imbalanced fraud data, accuracy alone can be misleading. Prioritize precision-recall, recall, F1, PR-AUC and the confusion matrix.")
        cm = np.array(metrics.get("confusion_matrix", [[0,0],[0,0]]))
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_xticks([0,1], labels=["Predicted Legitimate","Predicted Fraud"])
        ax.set_yticks([0,1], labels=["Actual Legitimate","Actual Fraud"])
        ax.set_title("Confusion matrix")
        for (i,j),v in np.ndenumerate(cm):
            ax.text(j,i,str(v),ha="center",va="center")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if metrics.get("classification_report"):
            st.markdown("#### Detailed classification report")
            st.json(metrics["classification_report"])
    else:
        st.info("Metrics will appear here after training. Run `python src/train_model.py --data data/creditcard.csv`.")

elif page == "About Project":
    st.subheader("Project summary")
    st.markdown("""
**Objective:** classify credit-card transactions as legitimate or potentially fraudulent.

**Classification techniques included**
- Logistic Regression
- Random Forest
- HistGradientBoosting (optional comparison baseline)

**Workflow**
1. Load and validate transaction data.
2. Split training and testing data using stratification.
3. Train multiple classifiers.
4. Compare precision, recall, F1-score, ROC-AUC and average precision.
5. Save the best preprocessing/model pipeline.
6. Predict and review transactions in this Streamlit dashboard.

**Dataset format:** a binary target column called `Class` (0 = legitimate, 1 = fraud) is supported. `Fraud` is also accepted.
""")
    st.warning("Educational project only. Do not use this model as the sole basis for blocking real payments. Protect cardholder data and restrict dashboard access.")
