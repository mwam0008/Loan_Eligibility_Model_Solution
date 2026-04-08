"""
model.py - ML logic for Loan Eligibility Prediction
Models: Logistic Regression, Decision Tree, Random Forest
With: Missing value imputation, MinMax scaling, cross-validation
"""

import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw credit CSV."""
    try:
        logging.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"Loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Load failed: {e}")
        raise


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline from the notebook:
    1. Drop Loan_ID
    2. Impute missing values
    3. One-hot encode categoricals
    4. Encode target Y/N → 1/0
    """
    try:
        df = df.copy()

        # Drop ID column
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)

        # Convert to object type (as in notebook)
        df['Credit_History'] = df['Credit_History'].astype('object')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')

        # Impute categorical missing values with mode
        for col in ['Gender', 'Married', 'Dependents', 'Self_Employed',
                    'Loan_Amount_Term', 'Credit_History']:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Impute numerical missing value with median (LoanAmount has outliers)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

        # One-hot encode categorical columns
        df = pd.get_dummies(
            df,
            columns=['Gender', 'Married', 'Dependents', 'Education',
                     'Self_Employed', 'Property_Area'],
            dtype=int
        )

        # Encode target
        # df['Loan_Approved'] = df['Loan_Approved'].replace({'Y': 1, 'N': 0})

        # Safety: fill any remaining NaNs after encoding
        df = df.fillna(0)

        # Encode target
        df['Loan_Approved'] = df['Loan_Approved'].replace({'Y': 1, 'N': 0})

        logging.info(f"Preprocessing complete. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


def split_and_scale(df: pd.DataFrame, test_size=0.2):
    """Split data and apply MinMax scaling."""
    try:
        # x = df.drop('Loan_Approved', axis=1)
        # y = df['Loan_Approved']

        x = df.drop('Loan_Approved', axis=1).fillna(0)
        y = df['Loan_Approved']

        xtrain, xtest, ytrain, ytest = train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=42
        )

        scale = MinMaxScaler()
        xtrain_scaled = scale.fit_transform(xtrain)
        xtest_scaled = scale.transform(xtest)

        logging.info(f"Train: {xtrain.shape}, Test: {xtest.shape}")
        return xtrain, xtest, ytrain, ytest, xtrain_scaled, xtest_scaled, scale, list(x.columns)
    except Exception as e:
        logging.error(f"Split/scale failed: {e}")
        raise


def train_logistic_regression(xtrain_scaled, ytrain):
    """Train Logistic Regression model."""
    try:
        logging.info("Training Logistic Regression...")
        model = LogisticRegression()
        model.fit(xtrain_scaled, ytrain)
        logging.info("LR trained.")
        return model
    except Exception as e:
        logging.error(f"LR training failed: {e}")
        raise


def train_decision_tree(xtrain_scaled, ytrain):
    """Train Decision Tree classifier."""
    try:
        logging.info("Training Decision Tree...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(xtrain_scaled, ytrain)
        logging.info("DT trained.")
        return model
    except Exception as e:
        logging.error(f"DT training failed: {e}")
        raise


def train_random_forest(xtrain_scaled, ytrain, n_estimators=100,
                        max_depth=None, max_features='sqrt'):
    """Train Random Forest classifier."""
    try:
        logging.info(f"Training Random Forest (n={n_estimators}, depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42
        )
        model.fit(xtrain_scaled, ytrain)
        logging.info("RF trained.")
        return model
    except Exception as e:
        logging.error(f"RF training failed: {e}")
        raise


def evaluate_model(model, xtest_scaled, ytest, threshold=0.5):
    """Return accuracy, confusion matrix, classification report."""
    try:
        if threshold != 0.5 and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(xtest_scaled)[:, 1]
            ypred = (proba >= threshold).astype(int)
        else:
            ypred = model.predict(xtest_scaled)

        acc = accuracy_score(ytest, ypred)
        cm = confusion_matrix(ytest, ypred)
        report = classification_report(ytest, ypred, output_dict=True)
        logging.info(f"Accuracy: {acc:.4f}")
        return acc, cm, report, ypred
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


def cross_validate(model, xtrain_scaled, ytrain, n_splits=5):
    """Run KFold cross-validation and return scores."""
    try:
        kfold = KFold(n_splits=n_splits)
        scores = cross_val_score(model, xtrain_scaled, ytrain, cv=kfold)
        logging.info(f"CV Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
        return scores
    except Exception as e:
        logging.error(f"Cross-validation failed: {e}")
        raise


def save_model(model, scaler, columns, path='Loan_Model.pkl'):
    """Save model, scaler, and column names together."""
    try:
        bundle = {'model': model, 'scaler': scaler, 'columns': columns}
        with open(path, 'wb') as f:
            pickle.dump(bundle, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Save failed: {e}")
        raise


def load_model(path='Loan_Model.pkl'):
    """Load model bundle."""
    try:
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        logging.info(f"Model loaded from {path}")
        return bundle['model'], bundle['scaler'], bundle['columns']
    except Exception as e:
        logging.error(f"Load failed: {e}")
        raise


def predict_single(model, scaler, columns, user_input: dict) -> tuple:
    """Predict loan eligibility for one applicant."""
    try:
        input_df = pd.DataFrame([user_input], columns=columns).fillna(0)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = float(prediction)
        return int(prediction), round(probability, 4)
    except Exception as e:
        logging.error(f"Single prediction failed: {e}")
        raise
