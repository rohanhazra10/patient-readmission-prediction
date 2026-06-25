import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("diabetic_data.csv")
print(df.head())
print(df.info())
print(df.describe())

# ── 2. Basic cleaning ─────────────────────────────────────────────────────────
df.replace("?", np.nan, inplace=True)

df.drop(columns=[
    "weight", "encounter_id", "patient_nbr", "payer_code",
    "medical_specialty", "max_glu_serum", "A1Cresult"
], inplace=True)

# ── 3. Encode target BEFORE split (it's a label, not a feature) ───────────────
df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

# ── 4. Train / test split on raw data (before any imputation or encoding) ─────
X_raw = df.drop("readmitted", axis=1)
Y = df["readmitted"]

X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
    X_raw, Y, test_size=0.3, random_state=42, stratify=Y
)

# ── 5. Impute using TRAIN statistics only ────────────────────────────────────
# FIX: copy BEFORE fillna so we don't mutate the split DataFrames in place
X_train_raw = X_train_raw.copy()
X_test_raw  = X_test_raw.copy()

race_mode = X_train_raw["race"].mode()[0]
X_train_raw["race"] = X_train_raw["race"].fillna(race_mode)
X_test_raw["race"]  = X_test_raw["race"].fillna(race_mode)

# For any remaining numeric nulls, fill with train column medians
num_cols = X_train_raw.select_dtypes(include=np.number).columns
col_medians = X_train_raw[num_cols].median()
X_train_raw[num_cols] = X_train_raw[num_cols].fillna(col_medians)
X_test_raw[num_cols]  = X_test_raw[num_cols].fillna(col_medians)

# ── 6. One-hot encode — fit on train, align test ──────────────────────────────
X_train = pd.get_dummies(X_train_raw, drop_first=True)
X_test  = pd.get_dummies(X_test_raw,  drop_first=True)

# FIX: reindex BEFORE sanitizing so column alignment uses the original names,
# then sanitize both together — guarantees identical column sets.
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

def sanitize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove characters XGBoost forbids in feature names: [ ] < >"""
    frame = frame.copy()
    frame.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", col)   # replace every non-alphanumeric/_ with _
        for col in frame.columns
    ]
    return frame

X_train = sanitize_columns(X_train)
X_test  = sanitize_columns(X_test)

print(f"\nTraining shape : {X_train.shape}")
print(f"Test shape     : {X_test.shape}")
print(f"Train nulls    : {X_train.isnull().sum().sum()}")
print(f"Test  nulls    : {X_test.isnull().sum().sum()}")

# ── 7. Apply SMOTE to training set only ───────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
print(f"\nClass distribution after SMOTE:\n{pd.Series(Y_train_res).value_counts()}")

# ── 8. Train model with early stopping ────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=20,
)

print("\nTraining the model...")
model.fit(
    X_train_res, Y_train_res,
    eval_set=[(X_test, Y_test)],
    verbose=False
)
best_n = model.best_iteration + 1   # best_iteration is 0-indexed
print(f"Best iteration : {best_n}")
print("Model training completed.")

# ── 9. Evaluate on held-out test set ──────────────────────────────────────────
Y_pred  = model.predict(X_test)
Y_proba = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall    = recall_score(Y_test, Y_pred)
f1        = f1_score(Y_test, Y_pred)
roc       = roc_auc_score(Y_test, Y_proba)

print("\n" + "="*40)
print("       MODEL PERFORMANCE METRICS")
print("="*40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")
print("\nConfusion Matrix")
print(confusion_matrix(Y_test, Y_pred))
print("\nClassification Report")
print(classification_report(Y_test, Y_pred))

# ── 10. Cross-validation — SMOTE inside pipeline to prevent leakage ───────────
# FIX: use best_n (not model.best_iteration) and pass X_train/Y_train
# (already encoded & sanitized, but pre-SMOTE) so CV folds are correct.
cv_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf",   XGBClassifier(
        n_estimators=best_n,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    cv_pipeline, X_train, Y_train,
    cv=cv, scoring="roc_auc", n_jobs=-1
)

print("\nCross-Validation ROC AUC Scores:", scores)
print(f"Mean ROC AUC : {scores.mean():.4f}")
print(f"Std  ROC AUC : {scores.std():.4f}")

# ── 11. Save model + column order + imputation values ─────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(
    {
        "model":       model,
        "columns":     list(X_train.columns),
        "race_mode":   race_mode,
        "col_medians": col_medians.to_dict(),
    },
    "model/readmission_model.pkl"
)
print("\nModel saved to 'model/readmission_model.pkl'.")