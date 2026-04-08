"""
app.py - Streamlit Web App for Loan Eligibility Prediction
Models: Logistic Regression, Decision Tree, Random Forest
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from model import (
    load_data, preprocess, split_and_scale,
    train_logistic_regression, train_decision_tree, train_random_forest,
    evaluate_model, cross_validate, save_model, load_model, predict_single,
)
from utils import (
    plot_loan_approval_distribution, plot_missing_values,
    plot_income_distribution, plot_confusion_matrix,
    plot_accuracy_comparison, plot_cross_validation,
    plot_feature_importance, plot_credit_history_approval,
)

# fix import name typo

st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

st.title("Loan Eligibility Prediction")
st.markdown("Predict whether a loan applicant should be **approved or denied** using ML classification models.")

DATA_PATH = "credit.csv"

@st.cache_data
def get_raw_data():
    return load_data(DATA_PATH)

try:
    raw_df = get_raw_data()
except Exception as e:
    st.error(f"Could not load credit.csv. Error: {e}")
    st.stop()

st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", [
    "Data Overview",
    "Train & Compare Models",
    "Cross Validation",
    "Predict Loan Eligibility",
])

# ════════════════════════════════════════════════════════════
# SECTION 1 - Data Overview
# ════════════════════════════════════════════════════════════
if section == "Data Overview":
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", raw_df.shape[0])
    col2.metric("Features", raw_df.shape[1] - 2)
    col3.metric("Approved", int((raw_df['Loan_Approved'] == 'Y').sum()))
    col4.metric("Denied", int((raw_df['Loan_Approved'] == 'N').sum()))

    st.subheader("Raw Data Sample")
    st.dataframe(raw_df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Loan Approval Distribution")
        fig = plot_loan_approval_distribution(raw_df)
        st.pyplot(fig)
    with col2:
        st.subheader("Missing Values")
        fig2 = plot_missing_values(raw_df)
        st.pyplot(fig2)

    st.subheader("Applicant Income by Loan Outcome")
    fig3 = plot_income_distribution(raw_df)
    st.pyplot(fig3)

    st.subheader("Credit History vs Loan Approval")
    st.markdown("Having a credit history dramatically increases approval chances.")
    fig4 = plot_credit_history_approval(raw_df)
    st.pyplot(fig4)

    st.subheader("Dataset Statistics")
    st.dataframe(raw_df.describe(include='all'))

# ════════════════════════════════════════════════════════════
# SECTION 2 - Train & Compare Models
# ════════════════════════════════════════════════════════════
elif section == "Train & Compare Models":
    st.header("Train & Compare All 3 Models")
    st.markdown("""
    The notebook trains **3 classifiers** and compares them.
    **Goal: accuracy above 76%.**
    """)

    st.sidebar.subheader("Settings")
    test_size    = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, step=0.05)
    n_estimators = st.sidebar.slider("RF: Number of Trees", 50, 300, 100, step=50)
    max_depth    = st.sidebar.select_slider("RF: Max Depth",
                       options=["None", 2, 3, 4, 5, 10], value="None")
    threshold    = st.sidebar.slider("LR: Decision Threshold", 0.3, 0.9, 0.5, step=0.05,
                       help="Default is 0.5. Increase to 0.7 to be more conservative.")

    md = None if max_depth == "None" else int(max_depth)

    if st.button("Train All 3 Models"):
        with st.spinner("Preprocessing data and training models..."):
            try:
                processed = preprocess(raw_df)
                xtrain, xtest, ytrain, ytest, xtrain_sc, xtest_sc, scaler, columns = \
                    split_and_scale(processed, test_size=test_size)

                # Train all 3
                lrmodel = train_logistic_regression(xtrain_sc, ytrain)
                dtmodel = train_decision_tree(xtrain_sc, ytrain)
                rfmodel = train_random_forest(xtrain_sc, ytrain,
                              n_estimators=n_estimators, max_depth=md)

                # Evaluate all 3
                lr_acc, lr_cm, lr_rep, _ = evaluate_model(lrmodel, xtest_sc, ytest, threshold)
                dt_acc, dt_cm, dt_rep, _ = evaluate_model(dtmodel, xtest_sc, ytest)
                rf_acc, rf_cm, rf_rep, _ = evaluate_model(rfmodel, xtest_sc, ytest)

                # Save best model
                best_model = rfmodel if rf_acc >= lr_acc and rf_acc >= dt_acc else \
                             lrmodel if lr_acc >= dt_acc else dtmodel
                save_model(best_model, scaler, columns, 'Loan_Model.pkl')

                st.success("All 3 models trained! Best model saved as Loan_Model.pkl")

                # Accuracy metrics
                st.subheader("Accuracy Comparison")
                results = {
                    "Logistic Regression": lr_acc,
                    "Decision Tree": dt_acc,
                    "Random Forest": rf_acc
                }

                col1, col2, col3 = st.columns(3)
                for col, (name, acc) in zip([col1, col2, col3], results.items()):
                    status = "sucess" if acc >= 0.76 else "fail"
                    col.metric(name, f"{acc:.2%}", delta=f"{status} {'Above' if acc >= 0.76 else 'Below'} 76% target")

                fig = plot_accuracy_comparison(results)
                st.pyplot(fig)

                # Confusion matrices
                st.subheader("Confusion Matrices")
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig = plot_confusion_matrix(lr_cm, "Logistic Regression")
                    st.pyplot(fig)
                with col2:
                    fig = plot_confusion_matrix(dt_cm, "Decision Tree")
                    st.pyplot(fig)
                with col3:
                    fig = plot_confusion_matrix(rf_cm, "Random Forest")
                    st.pyplot(fig)

                # Classification reports
                st.subheader("Classification Reports")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption("Logistic Regression")
                    st.dataframe(pd.DataFrame(lr_rep).T.round(3))
                with col2:
                    st.caption("Decision Tree")
                    st.dataframe(pd.DataFrame(dt_rep).T.round(3))
                with col3:
                    st.caption("Random Forest")
                    st.dataframe(pd.DataFrame(rf_rep).T.round(3))

                # Feature importance
                st.subheader("Random Forest Feature Importance")
                fig = plot_feature_importance(rfmodel, columns)
                st.pyplot(fig)

                # Store in session for CV tab
                st.session_state['lrmodel'] = lrmodel
                st.session_state['rfmodel'] = rfmodel
                st.session_state['xtrain_sc'] = xtrain_sc
                st.session_state['ytrain'] = ytrain

                winner = max(results, key=results.get)
                st.info(f"**{winner}** wins with {results[winner]:.2%} accuracy!")

            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# ════════════════════════════════════════════════════════════
# SECTION 3 - Cross Validation
# ════════════════════════════════════════════════════════════
elif section == "Cross Validation":
    st.header("Cross Validation")
    st.markdown("""
    **Cross Validation** tests the model on 5 different splits of the data,
    giving a more reliable accuracy estimate than a single train/test split.
    """)

    st.info("Please train models first in **Train & Compare Models** before running cross validation.")

    n_splits = st.slider("Number of Folds", 3, 10, 5)

    if st.button("Run Cross Validation"):
        if 'lrmodel' not in st.session_state:
            st.warning("Please train the models first!")
        else:
            with st.spinner("Running cross validation..."):
                try:
                    lrmodel = st.session_state['lrmodel']
                    rfmodel = st.session_state['rfmodel']
                    xtrain_sc = st.session_state['xtrain_sc']
                    ytrain = st.session_state['ytrain']

                    lr_scores = cross_validate(lrmodel, xtrain_sc, ytrain, n_splits)
                    rf_scores = cross_validate(rfmodel, xtrain_sc, ytrain, n_splits)

                    st.success("Cross validation complete!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Logistic Regression")
                        st.metric("Mean Accuracy", f"{lr_scores.mean():.2%}")
                        st.metric("Std Deviation", f"{lr_scores.std():.4f}")
                        fig = plot_cross_validation(lr_scores, "Logistic Regression")
                        st.pyplot(fig)

                    with col2:
                        st.subheader("Random Forest")
                        st.metric("Mean Accuracy", f"{rf_scores.mean():.2%}")
                        st.metric("Std Deviation", f"{rf_scores.std():.4f}")
                        fig = plot_cross_validation(rf_scores, "Random Forest")
                        st.pyplot(fig)

                    st.markdown("""
                    **How to read this:**
                    - Each bar = one fold's accuracy
                    - Blue dashed line = average across all folds
                    - Red dashed line = target (76%)
                    - Lower std deviation = more consistent model
                    """)

                except Exception as e:
                    st.error(f"Cross validation failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 4 - Predict Loan Eligibility
# ════════════════════════════════════════════════════════════
elif section == "Predict Loan Eligibility":
    st.header("Predict Loan Eligibility")
    st.markdown("Enter applicant details to predict if they should be approved.")

    try:
        model, scaler, columns = load_model('Loan_Model.pkl')
        model_available = True
        st.success("Trained model loaded!")
    except Exception:
        model_available = False
        st.warning("No trained model found. Please go to **Train & Compare Models** first!")

    if model_available:
        col1, col2 = st.columns(2)

        with col1:
            gender        = st.selectbox("Gender", ["Male", "Female"])
            married       = st.selectbox("Married", ["Yes", "No"])
            dependents    = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            applicant_income   = st.number_input("Applicant Income ($/month)", 0, 100000, 5000)
            coapplicant_income = st.number_input("Co-applicant Income ($/month)", 0, 50000, 0)
            loan_amount        = st.number_input("Loan Amount (x$1000)", 10, 1000, 150)
            loan_term          = st.selectbox("Loan Term (months)", [360, 120, 180, 240, 300, 480])
            credit_history     = st.selectbox("Credit History", [1, 0],
                                    format_func=lambda x: "Has Credit History" if x else "No Credit History")

        if st.button("Predict Eligibility"):
            try:
                # Build input matching one-hot encoded columns
                input_dict = {col: 0 for col in columns}

                # Fill numeric
                input_dict['ApplicantIncome']   = applicant_income
                input_dict['CoapplicantIncome'] = coapplicant_income
                input_dict['LoanAmount']        = loan_amount
                input_dict['Loan_Amount_Term']  = loan_term
                input_dict['Credit_History']    = credit_history

                # One-hot encoded fields
                input_dict[f'Gender_{gender}']              = 1
                input_dict[f'Married_{married}']            = 1
                input_dict[f'Dependents_{dependents}']      = 1
                input_dict[f'Education_{education}']        = 1
                input_dict[f'Self_Employed_{self_employed}']= 1
                input_dict[f'Property_Area_{property_area}']= 1

                # Keep only columns the model knows
                final_input = {col: input_dict.get(col, 0) for col in columns}

                prediction, probability = predict_single(model, scaler, columns, final_input)

                st.divider()
                if prediction == 1:
                    st.success(f"### LOAN APPROVED")
                    st.success(f"Approval Probability: **{probability:.2%}**")
                else:
                    st.error(f"### LOAN DENIED")
                    st.error(f"Approval Probability: **{probability:.2%}**")

                # Show probability gauge
                st.progress(probability)
                st.caption(f"Model confidence: {probability:.2%} chance of approval")

                with st.expander("Input Summary"):
                    st.dataframe(
                        pd.DataFrame([final_input]).T.rename(columns={0: 'Value'})
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.markdown("**Project**")
st.sidebar.markdown("Loan Eligibility Prediction")
