"""
logger.py - File-based activity logger for Loan Eligibility Prediction App
Writes all app events to app_activity.txt
"""

import logging
import os

LOG_FILE = "app_activity.txt"


# ── Configure file logger ─────────────────────────────────────
def _get_logger() -> logging.Logger:
    """Return a configured logger that writes to app_activity.txt."""
    logger = logging.getLogger("loan_eligibility_app")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


_logger = _get_logger()


# ── Public logging functions ──────────────────────────────────

def log_app_start() -> None:
    """Log when the Streamlit app starts up."""
    _logger.info("=" * 60)
    _logger.info("APP STARTED - Loan Eligibility Prediction")
    _logger.info("=" * 60)


def log_page_visit(page: str) -> None:
    """Log which section the user navigated to.

    Args:
        page: Name of the section visited.
    """
    _logger.info(f"PAGE VISIT        | section='{page}'")


def log_data_loaded(filepath: str, rows: int, cols: int) -> None:
    """Log successful data load.

    Args:
        filepath: Path to the CSV file loaded.
        rows: Number of rows in the dataset.
        cols: Number of columns in the dataset.
    """
    _logger.info(f"DATA LOADED       | file='{filepath}' rows={rows} cols={cols}")


def log_preprocessing(rows_before: int, rows_after: int, cols_after: int) -> None:
    """Log preprocessing pipeline completion.

    Args:
        rows_before: Row count before preprocessing.
        rows_after: Row count after preprocessing.
        cols_after: Column count after encoding.
    """
    _logger.info(
        f"PREPROCESSING     | rows={rows_before}→{rows_after} "
        f"cols_after_encoding={cols_after}"
    )


def log_model_training(model_name: str, test_size: float = None,
                        n_estimators: int = None, max_depth=None) -> None:
    """Log when a model training run starts.

    Args:
        model_name: Name of the model being trained.
        test_size: Fraction used for test split.
        n_estimators: Number of trees (Random Forest only).
        max_depth: Max tree depth (Random Forest only).
    """
    extras = ""
    if test_size is not None:
        extras += f" test_size={test_size}"
    if n_estimators is not None:
        extras += f" n_estimators={n_estimators}"
    if max_depth is not None:
        extras += f" max_depth={max_depth}"
    _logger.info(f"TRAINING STARTED  | model='{model_name}'{extras}")


def log_model_results(model_name: str, accuracy: float, target: float = 0.76) -> None:
    """Log model evaluation accuracy result.

    Args:
        model_name: Name of the evaluated model.
        accuracy: Test set accuracy (0.0 – 1.0).
        target: Accuracy goal threshold.
    """
    goal = "MET" if accuracy >= target else "NOT MET"
    _logger.info(
        f"MODEL RESULTS     | model='{model_name}' "
        f"accuracy={accuracy:.4f} ({accuracy:.2%}) "
        f"goal_{target:.0%}={goal}"
    )


def log_best_model_saved(model_name: str, accuracy: float, path: str) -> None:
    """Log when the best model is saved to disk.

    Args:
        model_name: Name of the winning model.
        accuracy: Its test accuracy.
        path: File path where it was saved.
    """
    _logger.info(
        f"MODEL SAVED       | best='{model_name}' "
        f"accuracy={accuracy:.2%} path='{path}'"
    )


def log_model_loaded(path: str) -> None:
    """Log when a model bundle is loaded from disk.

    Args:
        path: File path from which the model was loaded.
    """
    _logger.info(f"MODEL LOADED      | path='{path}'")


def log_cross_validation(model_name: str, n_splits: int,
                          mean_acc: float, std_acc: float) -> None:
    """Log cross-validation results.

    Args:
        model_name: Name of the model validated.
        n_splits: Number of CV folds used.
        mean_acc: Mean accuracy across folds.
        std_acc: Standard deviation of fold accuracies.
    """
    _logger.info(
        f"CROSS VALIDATION  | model='{model_name}' folds={n_splits} "
        f"mean={mean_acc:.2%} std={std_acc:.4f}"
    )


def log_prediction(applicant_summary: dict, prediction: int,
                   probability: float) -> None:
    """Log a loan eligibility prediction.

    Args:
        applicant_summary: Key fields describing the applicant.
        prediction: 1 = Approved, 0 = Denied.
        probability: Model's approval probability (0.0 – 1.0).
    """
    outcome = "APPROVED" if prediction == 1 else "DENIED"
    income  = applicant_summary.get("ApplicantIncome", "?")
    loan    = applicant_summary.get("LoanAmount", "?")
    credit  = applicant_summary.get("Credit_History", "?")
    area    = applicant_summary.get("property_area", "?")
    _logger.info(
        f"PREDICTION        | outcome={outcome} probability={probability:.2%} "
        f"income={income} loan_amount={loan} credit_history={credit} area={area}"
    )


def log_error(context: str, error: Exception) -> None:
    """Log an application error.

    Args:
        context: Description of where the error occurred.
        error: The exception that was raised.
    """
    _logger.error(
        f"ERROR             | context='{context}' "
        f"error={type(error).__name__}: {error}"
    )


def log_warning(message: str) -> None:
    """Log a non-critical warning.

    Args:
        message: Warning message to record.
    """
    _logger.warning(f"WARNING           | {message}")


def get_log_contents() -> str:
    """Read and return the full contents of the log file.

    Returns:
        Log file contents as a string, or a placeholder if not found.
    """
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No log file found yet. Activity will appear here after the app is used."
    except Exception as e:
        return f"Could not read log file: {e}"


def get_log_line_count() -> int:
    """Return the number of lines in the log file.

    Returns:
        Line count, or 0 if file not found.
    """
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
