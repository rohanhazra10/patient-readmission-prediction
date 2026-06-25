# 🏥 Diabetic Patient Readmission Predictor

A machine learning web application that predicts whether a diabetic patient will be readmitted to hospital within 30 days of discharge, using XGBoost with SMOTE oversampling and an interactive Streamlit interface.

---

## 📁 Project Structure

```
patient_readmission/
│
├── diabetic_data.csv            # Raw dataset (from UCI repository)
├── diabetic_readmission.py      # Standalone training script
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── model/
    └── readmission_model.pkl    # Saved model artifact (auto-created after training)
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the web app
```bash
streamlit run app.py
```

### 3. (Optional) Run training script standalone
```bash
python diabetic_readmission.py
```

> ⚠️ Note: You cannot run `python app.py` — Streamlit apps must always be launched with `streamlit run app.py`.

---

## 🗂️ Dataset

**Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)

**Size:** ~101,766 patient encounter records across 130 US hospitals over 10 years.

**Target variable:** `readmitted`
- Original values: `<30` (readmitted within 30 days), `>30` (readmitted after 30 days), `NO` (not readmitted)
- Encoded as binary: `1` = readmitted within 30 days, `0` = otherwise

### Columns Dropped
These columns were removed before training due to high missingness or irrelevance:

| Column | Reason Dropped |
|--------|---------------|
| `weight` | >96% missing values |
| `encounter_id` | Unique identifier, not a feature |
| `patient_nbr` | Unique identifier, not a feature |
| `payer_code` | >40% missing, not clinically relevant |
| `medical_specialty` | >50% missing |
| `max_glu_serum` | >95% missing |
| `A1Cresult` | >85% missing |

---

## 🧠 Machine Learning Pipeline

### Step-by-step breakdown

```
Raw CSV
  │
  ▼
Replace "?" with NaN → Drop irrelevant columns → Encode target (readmitted)
  │
  ▼
Train / Test Split (70% / 30%, stratified)
  │
  ├── Training set ──► Impute race (mode) + numeric cols (median)
  │                         │
  │                    get_dummies (one-hot encode)
  │                         │
  │                    Sanitize column names (remove [, ], <)
  │                         │
  │                    SMOTE (balance classes)
  │                         │
  │                    XGBClassifier.fit() + early stopping
  │
  └── Test set ──────► Same imputation + encoding (aligned to train columns)
                            │
                       Predict → Evaluate metrics
```

### Why each technique was used

| Technique | Why |
|-----------|-----|
| **Stratified split** | Preserves class ratio (~11% positive) in both train and test |
| **Impute from train only** | Prevents data leakage from test set into training statistics |
| **get_dummies on train, reindex test** | Ensures test never introduces unseen categories |
| **Column name sanitization** | XGBoost rejects names containing `[`, `]`, or `<` |
| **SMOTE** | Dataset is heavily imbalanced (~89% not readmitted); SMOTE creates synthetic minority samples only on training data |
| **Early stopping (20 rounds)** | Stops training when validation loss stops improving, prevents overfitting |
| **5-fold stratified CV** | SMOTE inside the pipeline per fold — no leakage into validation folds |

### Model: XGBClassifier

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `n_estimators` | 300 | Max number of trees (early stopping reduces this) |
| `learning_rate` | 0.05 | Step size shrinkage — lower = more robust, slower |
| `max_depth` | 6 | Max depth of each tree — controls complexity |
| `subsample` | 0.8 | 80% of rows sampled per tree — reduces overfitting |
| `colsample_bytree` | 0.8 | 80% of features sampled per tree |
| `eval_metric` | logloss | Binary cross-entropy — suitable for classification |
| `early_stopping_rounds` | 20 | Stop if no improvement for 20 consecutive rounds |

### What gets saved in the `.pkl` file

```python
{
    "model":       trained XGBClassifier,
    "columns":     list of feature names in exact training order,
    "race_mode":   most common race value (for imputing missing race),
    "col_medians": median of each numeric column (for imputing missing values)
}
```
This ensures that at inference time, the exact same preprocessing is applied.

---

## 🖥️ Web Application (app.py)

The Streamlit app has three pages, selectable from the sidebar.

### Sidebar

| Option | Description |
|--------|-------------|
| **Upload CSV** | Uploads `diabetic_data.csv`, trains the model from scratch, and saves the `.pkl` |
| **Use saved model (.pkl)** | Loads a previously trained `.pkl` — skips retraining, much faster |

---

### Page 1 — 🏠 Overview

Displays model performance metrics after training.

| Section | What it shows |
|---------|--------------|
| **KPI cards** | Accuracy, Precision, Recall, F1 Score, ROC AUC |
| **Confusion matrix** | True/False Positives and Negatives as a heatmap |
| **Feature importance chart** | Top 20 features ranked by XGBoost importance score |
| **Cross-validation chart** | ROC AUC score for each of the 5 CV folds + mean |
| **Classification report** | Per-class precision, recall, F1, support as a table |

**Understanding the metrics:**

| Metric | What it means | Good value |
|--------|--------------|------------|
| **Accuracy** | % of all predictions that were correct | >85% |
| **Precision** | Of patients flagged as high risk, how many actually were | >60% |
| **Recall** | Of actually high-risk patients, how many did we catch | >55% |
| **F1 Score** | Harmonic mean of precision and recall | >58% |
| **ROC AUC** | Model's ability to separate classes (1.0 = perfect) | >0.80 |

> In medical settings, **Recall** matters most — missing a high-risk patient (false negative) is more dangerous than a false alarm.

---

### Page 2 — 🔬 Single Prediction

A patient intake form where you fill in clinical details and get an individual readmission risk score.

#### Input Fields Explained

##### Demographics

| Field | Type | Options / Range | Meaning |
|-------|------|-----------------|---------|
| **Race** | Dropdown | Caucasian, AfricanAmerican, Hispanic, Asian, Other | Patient's reported race — used because readmission rates vary across demographics |
| **Gender** | Dropdown | Male, Female | Biological sex of the patient |
| **Age group** | Dropdown | [0-10) through [90-100) | Patient's age in 10-year brackets |
| **Admission Type** | Dropdown | 1–8 | How the patient was admitted (see table below) |
| **Discharge Disposition ID** | Number | 1–29 | Where the patient went after discharge (see table below) |
| **Admission Source ID** | Number | 1–25 | Where the patient came from before admission |

**Admission Type values:**
| ID | Meaning |
|----|---------|
| 1 | Emergency |
| 2 | Urgent |
| 3 | Elective |
| 4 | Newborn |
| 5 | Not Available |
| 7 | Trauma Center |

**Discharge Disposition ID (common values):**
| ID | Meaning |
|----|---------|
| 1 | Discharged to home |
| 2 | Discharged to care facility |
| 3 | Discharged to SNF (Skilled Nursing Facility) |
| 6 | Discharged to home with health service |
| 11 | Expired (deceased) |
| 18 | Not Available |

**Admission Source ID (common values):**
| ID | Meaning |
|----|---------|
| 1 | Physician referral |
| 2 | Clinic referral |
| 4 | Transfer from hospital |
| 7 | Emergency room |
| 17 | Walk-in |

---

##### Clinical Measurements

| Field | Range | Meaning |
|-------|-------|---------|
| **Time in hospital** | 1–14 days | Number of days the patient stayed. Longer stays often indicate more severe illness and higher readmission risk |
| **# Lab procedures** | 1–132 | Number of lab tests performed during the encounter. High counts suggest complex cases |
| **# Procedures** | 0–6 | Number of non-lab procedures (e.g. surgeries, imaging) |
| **# Medications** | 1–81 | Number of distinct medications prescribed. More medications often correlates with higher readmission risk |
| **# Outpatient visits** | 0–42 | Number of outpatient visits in the year before this encounter |
| **# Emergency visits** | 0–76 | Number of ER visits in the year before. A strong predictor — frequent ER use signals poor disease control |
| **# Inpatient visits** | 0–21 | Number of inpatient admissions in the year before. The single strongest predictor in most readmission models |

---

##### Diagnoses

ICD-9 diagnosis codes. You can enter a number or a code like `V30`.

| Field | Meaning |
|-------|---------|
| **Primary diagnosis (diag_1)** | The main reason for the hospital visit |
| **Secondary diagnosis (diag_2)** | Second most significant condition |
| **Additional diagnosis (diag_3)** | Third contributing condition |
| **# Diagnoses** | Total number of diagnoses entered in the system (1–16) |

**Common ICD-9 codes relevant to diabetic patients:**
| Code | Condition |
|------|-----------|
| 250 | Diabetes mellitus |
| 428 | Heart failure |
| 401 | Essential hypertension |
| 414 | Coronary artery disease |
| 585 | Chronic kidney disease |
| 272 | Disorders of lipid metabolism |
| 486 | Pneumonia |

---

##### Medications

Each of the 21 diabetes-related medications has a dropdown with four options:

| Value | Meaning |
|-------|---------|
| **No** | Medication was not prescribed |
| **Steady** | Medication prescribed, dosage unchanged |
| **Up** | Medication prescribed, dosage increased |
| **Down** | Medication prescribed, dosage decreased |

**Medications tracked:**

| Medication | Class |
|-----------|-------|
| Metformin | Biguanide |
| Repaglinide, Nateglinide | Meglitinides |
| Glimepiride, Glipizide, Glyburide | Sulfonylureas |
| Pioglitazone, Rosiglitazone | Thiazolidinediones |
| Insulin | Insulin |
| Acarbose, Miglitol | Alpha-glucosidase inhibitors |
| Chlorpropamide, Tolbutamide, Tolazamide, Acetohexamide | Older sulfonylureas |
| Troglitazone | Withdrawn thiazolidinedione |
| Glyburide-metformin, Glipizide-metformin, etc. | Combination pills |

---

##### Other Fields

| Field | Options | Meaning |
|-------|---------|---------|
| **Change in medication** | No / Ch | Whether any diabetes medication was changed during this encounter. `Ch` = changed |
| **Diabetes medication prescribed** | Yes / No | Whether any diabetes medication was prescribed at all during this encounter |

---

#### Output

After clicking **Predict Readmission Risk**:

| Output | Meaning |
|--------|---------|
| **HIGH RISK** | Model predicts probability ≥ 50% — patient likely to be readmitted within 30 days |
| **LOW RISK** | Model predicts probability < 50% — patient unlikely to be readmitted within 30 days |
| **Confidence %** | The raw probability score (e.g. 73.4% = model is 73.4% confident this patient will be readmitted) |
| **Gauge chart** | Visual representation of the probability from 0–100% |
| **Top contributing features** | The 5 features with highest XGBoost importance scores for this model overall |

---

### Page 3 — 📋 Batch Prediction

Upload a CSV file containing multiple patient records and score them all at once.

- The CSV must have the same column schema as the original `diabetic_data.csv`
- The app automatically preprocesses, encodes, and aligns columns
- Output includes `readmission_probability` and `predicted_label` columns appended to each row
- Results can be downloaded as a scored CSV
- A histogram shows the distribution of risk scores across all patients

---

## 🧪 Test Cases

Three example patients to verify model behaviour:

### Case 1 — High Risk 🔴 (expected >60%)
Elderly male, emergency admission, long stay, prior inpatient history, heart failure diagnosis.
```
Age: [70-80) | Gender: Male | Admission: Emergency | Time: 10 days
Lab procedures: 67 | Medications: 21 | Inpatient visits: 3
Primary diagnosis: 428 (heart failure) | Insulin: Up | Change: Ch
```

### Case 2 — Medium Risk 🟡 (expected 30–55%)
Middle-aged female, urgent admission, moderate stay, some prior history.
```
Age: [50-60) | Gender: Female | Admission: Urgent | Time: 4 days
Lab procedures: 45 | Medications: 14 | Inpatient visits: 1
Primary diagnosis: 250 (diabetes) | Insulin: Steady | Change: No
```

### Case 3 — Low Risk 🟢 (expected <20%)
Young female, elective admission, short stay, no prior hospital history.
```
Age: [30-40) | Gender: Female | Admission: Elective | Time: 2 days
Lab procedures: 28 | Medications: 8 | Inpatient visits: 0
Primary diagnosis: V30 | Insulin: No | Change: No
```

---

## 📊 Interpreting Results

### What makes a patient high risk?
Based on feature importance, the strongest predictors are:

1. **Number of inpatient visits** — prior hospitalizations are the biggest signal
2. **Number of medications** — more medications = more complex disease
3. **Time in hospital** — longer stays indicate severity
4. **Number of lab procedures** — more tests = sicker patient
5. **Number of diagnoses** — comorbidities increase risk
6. **Discharge disposition** — where a patient goes after discharge matters
7. **Insulin changes** — dosage adjustments suggest poor glucose control

### Threshold
The default decision threshold is **0.5** (50%). You can interpret the raw probability for nuanced decisions:
- `0.0 – 0.3` → Low risk, standard follow-up
- `0.3 – 0.5` → Moderate risk, consider enhanced monitoring
- `0.5 – 0.7` → High risk, schedule early follow-up
- `0.7 – 1.0` → Very high risk, consider extended stay or discharge to care facility

---

## ⚠️ Limitations

- Model trained on US hospital data from 1999–2008; may not reflect current clinical practice
- Does not account for medication dosage amounts, only whether they changed
- Binary outcome only — does not predict readmission after 30 days
- Should be used as a **decision support tool**, not a replacement for clinical judgement

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Train/test split, metrics, cross-validation |
| `xgboost` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE oversampling |
| `joblib` | Model serialization |
| `plotly` | Interactive charts |

---

## 👤 Author

Built as a clinical ML demonstration project using the UCI Diabetes 130-US Hospitals dataset."# patient-readmission-prediction" 
