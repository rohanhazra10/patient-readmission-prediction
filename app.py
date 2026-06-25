import re
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #1a2a4a 0%, #0d1b2e 100%);
    border: 1px solid #2a4a7a;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.header-icon { font-size: 3.5rem; }
.header-title { color: #e8f4fd; font-size: 2rem; font-weight: 700; margin: 0; }
.header-sub   { color: #7ea8cc; font-size: 0.95rem; margin: 0.3rem 0 0; }

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: #161b27;
    border: 1px solid #2a3a55;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-label { color: #7ea8cc; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }
.metric-value { color: #e8f4fd; font-size: 1.8rem; font-weight: 700; margin: .3rem 0 0; }

/* ── Risk badge ── */
.risk-high {
    background: linear-gradient(135deg,#4a1a1a,#2a0d0d);
    border: 1px solid #c0392b;
    border-radius: 12px; padding: 1.5rem; text-align: center; margin: 1rem 0;
}
.risk-low {
    background: linear-gradient(135deg,#0d2a1a,#091a10);
    border: 1px solid #27ae60;
    border-radius: 12px; padding: 1.5rem; text-align: center; margin: 1rem 0;
}
.risk-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: .1em; color: #aaa; }
.risk-value { font-size: 2.5rem; font-weight: 800; margin: .3rem 0; }
.risk-high .risk-value { color: #e74c3c; }
.risk-low  .risk-value { color: #2ecc71; }

/* ── Section headers ── */
.section-title {
    color: #7ea8cc;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .12em;
    margin: 1.5rem 0 .6rem;
    padding-bottom: .4rem;
    border-bottom: 1px solid #2a3a55;
}

/* ── Info box ── */
.info-box {
    background: #0d1b2e;
    border-left: 3px solid #3a7bd5;
    border-radius: 0 8px 8px 0;
    padding: .8rem 1rem;
    color: #7ea8cc;
    font-size: .85rem;
    margin-bottom: 1rem;
}

/* ── Uploaded file indicator ── */
.upload-ok {
    background:#0d2a1a; border:1px solid #27ae60;
    border-radius:8px; padding:.6rem 1rem;
    color:#2ecc71; font-size:.85rem; margin-bottom:.8rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
DROP_COLS = ["weight","encounter_id","patient_nbr","payer_code",
             "medical_specialty","max_glu_serum","A1Cresult"]

def sanitize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in frame.columns]
    return frame

@st.cache_resource(show_spinner=False)
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df.replace("?", np.nan, inplace=True)
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    X_raw = df.drop("readmitted", axis=1)
    Y     = df["readmitted"]

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_raw, Y, test_size=0.3, random_state=42, stratify=Y)

    X_tr, X_te = X_tr.copy(), X_te.copy()

    race_mode = X_tr["race"].mode()[0] if "race" in X_tr.columns else None
    if race_mode:
        X_tr["race"] = X_tr["race"].fillna(race_mode)
        X_te["race"] = X_te["race"].fillna(race_mode)

    num_cols    = X_tr.select_dtypes(include=np.number).columns
    col_medians = X_tr[num_cols].median()
    X_tr[num_cols] = X_tr[num_cols].fillna(col_medians)
    X_te[num_cols] = X_te[num_cols].fillna(col_medians)

    X_tr = pd.get_dummies(X_tr, drop_first=True)
    X_te = pd.get_dummies(X_te, drop_first=True)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    X_tr = sanitize_columns(X_tr)
    X_te = sanitize_columns(X_te)

    smote = SMOTE(random_state=42)
    X_res, Y_res = smote.fit_resample(X_tr, Y_tr)

    mdl = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", early_stopping_rounds=20,
    )
    mdl.fit(X_res, Y_res, eval_set=[(X_te, Y_te)], verbose=False)
    best_n = mdl.best_iteration + 1

    Y_pred  = mdl.predict(X_te)
    Y_proba = mdl.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(Y_te, Y_pred),
        "precision": precision_score(Y_te, Y_pred),
        "recall":    recall_score(Y_te, Y_pred),
        "f1":        f1_score(Y_te, Y_pred),
        "roc_auc":   roc_auc_score(Y_te, Y_proba),
        "cm":        confusion_matrix(Y_te, Y_pred),
        "report":    classification_report(Y_te, Y_pred, output_dict=True),
        "best_n":    best_n,
    }

    # CV
    cv_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   XGBClassifier(
            n_estimators=best_n, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss",
        ))
    ])
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(cv_pipe, X_tr, Y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    metrics["cv_scores"] = scores

    # Feature importance
    fi = pd.Series(mdl.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    metrics["feature_importance"] = fi

    artifact = {
        "model":       mdl,
        "columns":     list(X_tr.columns),
        "race_mode":   race_mode,
        "col_medians": col_medians.to_dict(),
    }
    return artifact, metrics, df

def preprocess_single(row: dict, artifact: dict) -> pd.DataFrame:
    df_row = pd.DataFrame([row])
    df_row.replace("?", np.nan, inplace=True)

    if artifact["race_mode"] and "race" in df_row.columns:
        df_row["race"] = df_row["race"].fillna(artifact["race_mode"])

    num_cols = df_row.select_dtypes(include=np.number).columns
    for c in num_cols:
        if df_row[c].isnull().any() and c in artifact["col_medians"]:
            df_row[c] = df_row[c].fillna(artifact["col_medians"][c])

    df_row = pd.get_dummies(df_row, drop_first=True)
    df_row = df_row.reindex(columns=pd.Index(artifact["columns"]).map(
        lambda x: x), fill_value=0)

    # rebuild with sanitized names
    df_row = sanitize_columns(df_row)
    # align again after sanitize
    target_cols = artifact["columns"]
    df_row = df_row.reindex(columns=target_cols, fill_value=0)
    return df_row

def preprocess_batch(df_raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    df = df_raw.copy()
    df.replace("?", np.nan, inplace=True)
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=existing_drop, errors="ignore", inplace=True)
    if "readmitted" in df.columns:
        df.drop(columns=["readmitted"], inplace=True)

    if artifact["race_mode"] and "race" in df.columns:
        df["race"] = df["race"].fillna(artifact["race_mode"])

    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        if c in artifact["col_medians"]:
            df[c] = df[c].fillna(artifact["col_medians"][c])

    df = pd.get_dummies(df, drop_first=True)
    df = sanitize_columns(df)
    df = df.reindex(columns=artifact["columns"], fill_value=0)
    return df

def gauge_chart(prob: float) -> go.Figure:
    color = "#e74c3c" if prob >= 0.5 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#7ea8cc",
                      "tickfont": {"color": "#7ea8cc"}},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "#161b27",
            "bordercolor": "#2a3a55",
            "steps": [
                {"range": [0,  50], "color": "#0d2a1a"},
                {"range": [50, 100], "color": "#2a0d0d"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.75, "value": prob * 100},
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=0, l=30, r=30),
        paper_bgcolor="#0f1117", font_color="#e8f4fd",
    )
    return fig

def confusion_matrix_fig(cm: np.ndarray) -> go.Figure:
    labels = ["Not Readmitted", "Readmitted <30d"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,"#0d1b2e"],[1,"#3a7bd5"]],
        text=cm, texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        showscale=False,
    ))
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        height=300, margin=dict(t=10, b=60, l=80, r=10),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#e8f4fd",
        xaxis=dict(tickfont=dict(color="#7ea8cc")),
        yaxis=dict(tickfont=dict(color="#7ea8cc")),
    )
    return fig

def fi_chart(fi: pd.Series, top_n: int = 20) -> go.Figure:
    top = fi.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index,
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale=[[0,"#1a3a5c"],[1,"#3a7bd5"]],
            line=dict(color="#2a3a55", width=0.5),
        ),
    ))
    fig.update_layout(
        height=500, margin=dict(t=10, b=40, l=10, r=10),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#e8f4fd",
        xaxis=dict(title="Importance", gridcolor="#1e2a3a",
                   tickfont=dict(color="#7ea8cc")),
        yaxis=dict(tickfont=dict(color="#7ea8cc")),
    )
    return fig

def cv_chart(scores: np.ndarray) -> go.Figure:
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=folds, y=scores,
        marker_color="#3a7bd5",
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont=dict(color="#7ea8cc", size=11),
    ))
    fig.add_hline(y=scores.mean(), line_dash="dash",
                  line_color="#e74c3c", annotation_text=f"Mean {scores.mean():.4f}",
                  annotation_font_color="#e74c3c")
    fig.update_layout(
        height=280, margin=dict(t=20, b=40, l=10, r=10),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#e8f4fd",
        yaxis=dict(range=[0, 1], gridcolor="#1e2a3a",
                   tickfont=dict(color="#7ea8cc")),
        xaxis=dict(tickfont=dict(color="#7ea8cc")),
    )
    return fig

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">📂 Data Source</p>', unsafe_allow_html=True)
    data_source = st.radio("", ["Upload CSV", "Use saved model (.pkl)"],
                           label_visibility="collapsed")

    artifact, metrics, raw_df = None, None, None

    if data_source == "Upload CSV":
        uploaded_csv = st.file_uploader("diabetic_data.csv", type=["csv"])
        if uploaded_csv:
            tmp = "/tmp/diabetic_data_upload.csv"
            with open(tmp, "wb") as f:
                f.write(uploaded_csv.read())
            st.markdown('<div class="upload-ok">✅ CSV loaded — training model…</div>',
                        unsafe_allow_html=True)
            with st.spinner("Training XGBoost model…"):
                artifact, metrics, raw_df = train_model(tmp)
            st.success(f"Done! Best iteration: {metrics['best_n']}")

            save_path = "model/readmission_model.pkl"
            os.makedirs("model", exist_ok=True)
            joblib.dump(artifact, save_path)
            with open(save_path, "rb") as f:
                st.download_button("⬇ Download model (.pkl)", f,
                                   file_name="readmission_model.pkl",
                                   mime="application/octet-stream")
    else:
        uploaded_pkl = st.file_uploader("readmission_model.pkl", type=["pkl"])
        if uploaded_pkl:
            artifact = joblib.load(uploaded_pkl)
            st.success("✅ Model loaded")

    st.markdown('<p class="section-title">🔀 Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 Overview", "🔬 Single Prediction", "📋 Batch Prediction"],
                    label_visibility="collapsed")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div class="header-icon">🏥</div>
  <div>
    <p class="header-title">Diabetic Patient Readmission Predictor</p>
    <p class="header-sub">XGBoost · SMOTE · Early Stopping · 5-Fold Cross Validation</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

# ── OVERVIEW ──────────────────────────────────
if page == "🏠 Overview":
    if metrics is None:
        st.markdown('<div class="info-box">👈 Upload <b>diabetic_data.csv</b> from the sidebar to train the model and see performance metrics.</div>',
                    unsafe_allow_html=True)
        st.stop()

    # KPI cards
    m = metrics
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{m['accuracy']:.3f}</div></div>
      <div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value">{m['precision']:.3f}</div></div>
      <div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value">{m['recall']:.3f}</div></div>
      <div class="metric-card"><div class="metric-label">F1 Score</div><div class="metric-value">{m['f1']:.3f}</div></div>
      <div class="metric-card"><div class="metric-label">ROC AUC</div><div class="metric-value">{m['roc_auc']:.3f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        st.plotly_chart(confusion_matrix_fig(m["cm"]), use_container_width=True)

        st.markdown('<p class="section-title">Cross-Validation ROC AUC (5 Folds)</p>', unsafe_allow_html=True)
        st.plotly_chart(cv_chart(m["cv_scores"]), use_container_width=True)
        cv = m["cv_scores"]
        st.markdown(f'<div class="info-box">Mean: <b>{cv.mean():.4f}</b> &nbsp;|&nbsp; Std: <b>{cv.std():.4f}</b> &nbsp;|&nbsp; Best iteration: <b>{m["best_n"]}</b></div>',
                    unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">Top 20 Feature Importances</p>', unsafe_allow_html=True)
        st.plotly_chart(fi_chart(m["feature_importance"]), use_container_width=True)

    st.markdown('<p class="section-title">Classification Report</p>', unsafe_allow_html=True)
    report_df = pd.DataFrame(m["report"]).T.round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues", axis=None)
                 .format(precision=3), use_container_width=True)

# ── SINGLE PREDICTION ─────────────────────────
elif page == "🔬 Single Prediction":
    if artifact is None:
        st.markdown('<div class="info-box">👈 Upload <b>diabetic_data.csv</b> to train, or load a saved <b>.pkl</b> model.</div>',
                    unsafe_allow_html=True)
        st.stop()

    st.markdown('<p class="section-title">Patient Demographics</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        race = st.selectbox("Race", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other"])
        gender = st.selectbox("Gender", ["Male","Female"])
    with c2:
        age = st.selectbox("Age Group", [
            "[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
            "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"])
        admission_type_id = st.selectbox("Admission Type", [1,2,3,4,5,6,7,8],
            format_func=lambda x: {1:"Emergency",2:"Urgent",3:"Elective",
                                    4:"Newborn",5:"Not Available",6:"NULL",
                                    7:"Trauma Center",8:"Not Mapped"}[x])
    with c3:
        discharge_disposition_id = st.number_input("Discharge Disposition ID", 1, 29, 1)
        admission_source_id      = st.number_input("Admission Source ID", 1, 25, 7)

    st.markdown('<p class="section-title">Clinical Measurements</p>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        time_in_hospital  = st.slider("Time in Hospital (days)", 1, 14, 3)
        num_lab_procedures = st.slider("# Lab Procedures", 1, 132, 40)
    with c5:
        num_procedures    = st.slider("# Procedures", 0, 6, 1)
        num_medications   = st.slider("# Medications", 1, 81, 15)
    with c6:
        number_outpatient = st.slider("# Outpatient Visits", 0, 42, 0)
        number_emergency  = st.slider("# Emergency Visits", 0, 76, 0)
        number_inpatient  = st.slider("# Inpatient Visits", 0, 21, 0)

    st.markdown('<p class="section-title">Diagnoses</p>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        diag_1 = st.text_input("Primary Diagnosis (ICD-9)", "428")
    with c8:
        diag_2 = st.text_input("Secondary Diagnosis", "250")
    with c9:
        diag_3 = st.text_input("Additional Diagnosis", "401")

    number_diagnoses = st.slider("# Diagnoses", 1, 16, 5)

    st.markdown('<p class="section-title">Medications</p>', unsafe_allow_html=True)
    med_cols = st.columns(5)
    meds = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
            "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
            "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
            "insulin","glyburide_metformin","glipizide_metformin",
            "glimepiride_pioglitazone","metformin_rosiglitazone","metformin_pioglitazone"]
    med_values = {}
    for i, med in enumerate(meds):
        with med_cols[i % 5]:
            med_values[med] = st.selectbox(med.replace("_"," ").title(),
                                           ["No","Steady","Up","Down"], key=f"med_{med}")

    change   = st.selectbox("Change in Medication", ["No","Ch"])
    diabetesMed = st.selectbox("Diabetes Medication Prescribed", ["Yes","No"])

    row = {
        "race": race, "gender": gender, "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
        "change": change, "diabetesMed": diabetesMed,
        **med_values,
    }

    if st.button("🔍 Predict Readmission Risk", use_container_width=True):
        X_input = preprocess_single(row, artifact)
        prob    = artifact["model"].predict_proba(X_input)[0, 1]
        label   = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
        css_cls = "risk-high" if prob >= 0.5 else "risk-low"

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div class="{css_cls}">
              <div class="risk-label">Readmission within 30 days</div>
              <div class="risk-value">{label}</div>
              <div style="color:#aaa;font-size:.85rem;">Confidence: {prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)

        st.markdown('<p class="section-title">Top contributing features</p>', unsafe_allow_html=True)
        fi  = artifact["model"].feature_importances_
        cols = artifact["columns"]
        top5 = sorted(zip(cols, fi), key=lambda x: -x[1])[:5]
        for feat, imp in top5:
            st.progress(float(imp / max(fi)), text=f"{feat}  ({imp:.4f})")

# ── BATCH PREDICTION ──────────────────────────
elif page == "📋 Batch Prediction":
    if artifact is None:
        st.markdown('<div class="info-box">👈 Upload <b>diabetic_data.csv</b> to train, or load a saved <b>.pkl</b> model.</div>',
                    unsafe_allow_html=True)
        st.stop()

    st.markdown('<p class="section-title">Upload patient records for batch scoring</p>',
                unsafe_allow_html=True)
    batch_file = st.file_uploader("CSV with patient records (same schema as training data)",
                                  type=["csv"], key="batch")

    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.markdown(f'<div class="info-box">Loaded <b>{len(batch_df)}</b> records · <b>{batch_df.shape[1]}</b> columns</div>',
                    unsafe_allow_html=True)

        with st.spinner("Scoring records…"):
            X_batch = preprocess_batch(batch_df, artifact)
            probs   = artifact["model"].predict_proba(X_batch)[:, 1]
            preds   = (probs >= 0.5).astype(int)

        results = batch_df.copy()
        results["readmission_probability"] = probs.round(4)
        results["predicted_label"]         = np.where(preds == 1, "HIGH RISK", "LOW RISK")

        # Summary KPIs
        high = int(preds.sum())
        low  = len(preds) - high
        avg  = float(probs.mean())

        k1, k2, k3 = st.columns(3)
        k1.metric("🔴 High Risk Patients", high)
        k2.metric("🟢 Low Risk Patients",  low)
        k3.metric("📊 Avg Risk Score",     f"{avg:.3f}")

        # Distribution chart
        st.markdown('<p class="section-title">Risk Score Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            x=probs, nbins=40, color_discrete_sequence=["#3a7bd5"],
            labels={"x": "Readmission Probability"},
        )
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c",
                           annotation_text="Threshold 0.5",
                           annotation_font_color="#e74c3c")
        fig_hist.update_layout(
            height=280, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#e8f4fd", margin=dict(t=10, b=40, l=10, r=10),
            yaxis=dict(gridcolor="#1e2a3a"),
            xaxis=dict(tickfont=dict(color="#7ea8cc")),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<p class="section-title">Scored Records</p>', unsafe_allow_html=True)
        st.dataframe(
            results[["readmission_probability","predicted_label"] +
                    [c for c in results.columns
                     if c not in ("readmission_probability","predicted_label")]
                   ].style.background_gradient(
                        subset=["readmission_probability"], cmap="RdYlGn_r"),
            use_container_width=True, height=400,
        )

        csv_out = results.to_csv(index=False).encode()
        st.download_button("⬇ Download scored CSV", csv_out,
                           file_name="scored_patients.csv", mime="text/csv",
                           use_container_width=True)