from pathlib import Path

import numpy as np

from helpers import (
    create_csv_submission,
    find_best_threshold,
    fit_preprocessor,
    load_csv_data,
    transform_features,
)
from implementations import ridge_regression


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

    medians_val, means_val, stds_val, missing_columns_val = fit_preprocessor(
        x_train[tr]
    )

    x_tr = transform_features(
        x_train[tr], medians_val, means_val, stds_val, missing_columns_val
    )

    x_val = transform_features(
        x_train[va], medians_val, means_val, stds_val, missing_columns_val
    )

    # Compare several regularization strengths using the same validation split.
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    best_lambda = None
    best_threshold = None
    best_f1 = -1.0
    best_accuracy = 0.0
    scores_val = None

    print("\nLambda search:")
    for lambda_ in lambda_values:
        candidate_weights, _ = ridge_regression(y_train[tr], x_tr, lambda_=lambda_)
        candidate_scores = x_val @ candidate_weights
        candidate_threshold, candidate_f1 = find_best_threshold(
            y_train[va], candidate_scores
        )
        candidate_predictions = np.where(candidate_scores >= candidate_threshold, 1, -1)
        candidate_accuracy = np.mean(candidate_predictions == y_train[va])

        print(
            f"lambda={lambda_:g}, threshold={candidate_threshold:.6f}, "
            f"accuracy={candidate_accuracy:.6f}, F1={candidate_f1:.6f}"
        )

        if candidate_f1 > best_f1:
            best_lambda = lambda_
            best_threshold = candidate_threshold
            best_f1 = candidate_f1
            best_accuracy = candidate_accuracy
            scores_val = candidate_scores

    pred_val = np.where(scores_val >= best_threshold, 1, -1)

    print("\nSelected lambda:", best_lambda)
    print("Best threshold:", best_threshold)
    print("Val accuracy:", best_accuracy)
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

    # Release validation arrays before processing the complete dataset.
    del x_tr, x_val, scores_val

    medians_final, means_final, stds_final, missing_columns_final = fit_preprocessor(
        x_train
    )

    x_train_final = transform_features(
        x_train, medians_final, means_final, stds_final, missing_columns_final
    )
    x_test_final = transform_features(
        x_test, medians_final, means_final, stds_final, missing_columns_final
    )

    # Train a basic ridge-regression model.
    weights, loss = ridge_regression(y_train, x_train_final, lambda_=best_lambda)

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
