# Loan Eligibility Prediction

A Streamlit web app that predicts whether a loan applicant should be **approved or denied** using 3 classification models.

## What This App Does

| Section | What it shows |
|---|---|
| Data Overview | Approval distribution, missing values, income analysis, credit history impact |
| Train & Compare Models | Train LR + DT + RF, compare accuracy, confusion matrices, feature importance |
| Cross Validation | 5-fold cross validation for LR and RF |
| Predict Loan Eligibility | Enter applicant details → get approved/denied prediction |

## Goal

**Accuracy above 76%** on the test set.

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/loan-eligibility-predictor.git
cd loan-eligibility-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
loan_app/
├── app.py            ← Streamlit web app (4 pages)
├── model.py          ← All ML logic: preprocessing, training, evaluation, pickle
├── utils.py          ← All charts and visualizations
├── credit.csv        ← Credit dataset (614 applicants)
├── requirements.txt  ← Dependencies
└── README.md         ← This file
```

## Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Baseline classifier, probability threshold tunable |
| Decision Tree | Interpretable, can overfit |
| Random Forest | Best accuracy, ensemble of trees |

## Key Concepts

- **Missing Value Imputation** - mode for categorical, median for numerical
- **One-Hot Encoding** - converts categorical columns to numbers
- **MinMax Scaling** - normalizes features to 0-1 range
- **Cross Validation** - KFold with 5 splits for reliable accuracy estimate
- **Decision Threshold** - adjust LR threshold from 0.5 to 0.7 for conservative approval

## Dataset

614 Credit loan applicants with 13 features including income, loan amount, credit history, and property area.
