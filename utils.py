"""
utils.py - Visualization helpers for Loan Eligibility App
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)


def plot_loan_approval_distribution(df: pd.DataFrame):
    """Bar chart of approved vs denied loans."""
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['Loan_Approved'].value_counts()
        colors = ['#4CAF50', '#F44336']
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold')
        ax.set_title('Loan Approval Distribution', fontweight='bold')
        ax.set_xlabel('Loan Approved (Y = Yes, N = No)')
        ax.set_ylabel('Count')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Distribution plot error: {e}")
        raise


def plot_missing_values(df: pd.DataFrame):
    """Bar chart showing missing values per column."""
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(missing.index, missing.values, color='#FF9800', edgecolor='white')
        for bar, val in zip(bars, missing.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', fontweight='bold')
        ax.set_title('Missing Values Per Column', fontweight='bold')
        ax.set_ylabel('Missing Count')
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Missing values plot error: {e}")
        raise


def plot_income_distribution(df: pd.DataFrame):
    """Histogram of applicant income by loan approval."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        approved = df[df['Loan_Approved'] == 'Y']['ApplicantIncome']
        denied = df[df['Loan_Approved'] == 'N']['ApplicantIncome']
        ax.hist(approved, bins=30, alpha=0.6, color='#4CAF50', label='Approved')
        ax.hist(denied, bins=30, alpha=0.6, color='#F44336', label='Denied')
        ax.set_title('Applicant Income by Loan Outcome', fontweight='bold')
        ax.set_xlabel('Income')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Income dist plot error: {e}")
        raise


def plot_confusion_matrix(cm, model_name: str):
    """Heatmap of confusion matrix."""
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Denied (0)', 'Approved (1)'],
                    yticklabels=['Denied (0)', 'Approved (1)'], ax=ax)
        ax.set_title(f'{model_name} — Confusion Matrix', fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Confusion matrix error: {e}")
        raise


def plot_accuracy_comparison(results: dict):
    """Bar chart comparing accuracy of all models."""
    try:
        models = list(results.keys())
        accs = [results[m] for m in models]
        colors = ['#2196F3', '#FF9800', '#4CAF50']

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(models, accs, color=colors[:len(models)], edgecolor='white')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.2%}', ha='center', fontweight='bold')
        ax.axhline(0.76, color='red', linestyle='--', label='Target Accuracy (76%)')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Accuracy comparison error: {e}")
        raise


def plot_cross_validation(scores: np.ndarray, model_name: str):
    """Bar chart of cross-validation fold scores."""
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        folds = [f'Fold {i+1}' for i in range(len(scores))]
        colors = ['#4CAF50' if s >= 0.76 else '#FF9800' for s in scores]
        bars = ax.bar(folds, scores, color=colors, edgecolor='white')
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{s:.2%}', ha='center', fontweight='bold', fontsize=9)
        ax.axhline(scores.mean(), color='blue', linestyle='--',
                   label=f'Mean: {scores.mean():.2%}')
        ax.axhline(0.76, color='red', linestyle='--', label='Target (76%)')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name} - Cross Validation (5 Folds)', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"CV plot error: {e}")
        raise


def plot_feature_importance(model, columns: list):
    """Horizontal bar of Random Forest feature importances."""
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), importances[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([columns[i] for i in indices], fontsize=8)
        ax.set_xlabel('Importance Score')
        ax.set_title('Random Forest - Feature Importance', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Feature importance error: {e}")
        raise


def plot_credit_history_approval(df: pd.DataFrame):
    """Stacked bar: loan approval rate by credit history."""
    try:
        ct = pd.crosstab(df['Credit_History'], df['Loan_Approved'], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        ct.plot(kind='bar', stacked=True, color=['#F44336', '#4CAF50'], ax=ax, edgecolor='white')
        ax.set_title('Loan Approval Rate by Credit History', fontweight='bold')
        ax.set_xlabel('Credit History (1 = Has history, 0 = No history)')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Denied', 'Approved'])
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Credit history plot error: {e}")
        raise
