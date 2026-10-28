from pathlib import Path

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import ridge_regression


def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def fit_preprocessor(x):
    """Learn preprocessing values from training data."""
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)

    x_filled = np.where(np.isnan(x), medians, x)

    means = np.mean(x_filled, axis=0)
    standard_deviations = np.std(x_filled, axis=0)
    standard_deviations[standard_deviations == 0] = 1

    return medians, means, standard_deviations


def transform_features(x, medians, means, standard_deviations):
    """Apply previously learned preprocessing values."""
    x = np.where(np.isnan(x), medians, x)
    x = (x - means) / standard_deviations

    # Add the bias column.
    return np.column_stack((np.ones(len(x)), x))


def main():
    # Locate the data folder relative to this file.
    project_directory = Path(__file__).resolve().parent
    data_directory = project_directory / "data"

    # Load the dataset.
    x_train, x_test, y_train, _, test_ids = load_csv_data(data_directory)

    print("Training shape:", x_train.shape)
    print("Test shape:", x_test.shape)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_train))

    cut = int(0.8 * len(idx))
    tr, va = idx[:cut], idx[cut:]

    medians_val, means_val, stds_val = fit_preprocessor(x_train[tr])

    x_tr = transform_features(x_train[tr], medians_val, means_val, stds_val)

    x_val = transform_features(x_train[va], medians_val, means_val, stds_val)

    w_val, _ = ridge_regression(y_train[tr], x_tr, lambda_=0.001)
    scores_val = x_val @ w_val

    best_threshold = 0.0
    best_f1 = -1.0

    # Try thresholds corresponding to different proportions
    # of positive predictions.
    thresholds = np.quantile(
        scores_val,
        np.linspace(0.50, 0.99, 200),
    )

    for threshold in thresholds:
        candidate_predictions = np.where(scores_val >= threshold, 1, -1)
        candidate_f1 = f1_score(y_train[va], candidate_predictions)

        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_threshold = threshold

    pred_val = np.where(scores_val >= best_threshold, 1, -1)

    print("Best threshold:", best_threshold)
    print("Val accuracy:", np.mean(pred_val == y_train[va]))
    print("Val F1:", best_f1)

    y_val = y_train[va]

    tp = np.sum((y_val == 1) & (pred_val == 1))
    tn = np.sum((y_val == -1) & (pred_val == -1))
    fp = np.sum((y_val == -1) & (pred_val == 1))
    fn = np.sum((y_val == 1) & (pred_val == -1))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    print("True positives:", tp)
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("Precision:", precision)
    print("Recall:", recall)
    medians_final, means_final, stds_final = fit_preprocessor(x_train)

    x_train_final = transform_features(x_train, medians_final, means_final, stds_final)
    x_test_final = transform_features(x_test, medians_final, means_final, stds_final)

    # Train a basic ridge-regression model.
    weights, loss = ridge_regression(y_train, x_train_final, lambda_=0.001)

    print("Training loss:", loss)

    # Convert continuous scores into {-1, 1} predictions.
    scores = x_test_final @ weights
    predictions = np.where(scores >= best_threshold, 1, -1)

    # Produce the AIcrowd submission.
    output_path = project_directory / "submission.csv"
    create_csv_submission(test_ids, predictions, output_path)

    print("Submission created:", output_path)


if __name__ == "__main__":
    main()
