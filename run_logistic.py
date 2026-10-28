from pathlib import Path

import numpy as np

from helpers import (
    create_csv_submission,
    find_best_threshold,
    fit_preprocessor,
    load_csv_data,
    transform_features,
)
from implementations import reg_logistic_regression


def main():
    # Locate the data folder relative to this file.
    project_directory = Path(__file__).resolve().parent
    data_directory = project_directory / "data"

    # Load the dataset.
    x_train, x_test, y_train, _, test_ids = load_csv_data(data_directory)
    # Convert labels from {-1, 1} to {0, 1} for logistic regression.
    y_train_01 = (y_train + 1) / 2

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

    # Initial logistic-regression configuration.
    lambda_ = 0.00001
    max_iters = 300
    gamma = 0.1

    initial_w = np.zeros(x_tr.shape[1])

    w_val, validation_training_loss = reg_logistic_regression(
        y_train_01[tr],
        x_tr,
        lambda_=lambda_,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
    )

    scores_val = x_val @ w_val

    best_threshold, best_f1 = find_best_threshold(y_train[va], scores_val)

    pred_val = np.where(scores_val >= best_threshold, 1, -1)

    best_accuracy = np.mean(pred_val == y_train[va])

    print("Lambda:", lambda_)
    print("Learning rate:", gamma)
    print("Iterations:", max_iters)
    print("Best threshold:", best_threshold)
    print("Val accuracy:", best_accuracy)
    print("Val F1:", best_f1)
    print("Validation-training loss:", validation_training_loss)

    # Check if the logistic model beats the ridge baseline F1 score.
    ridge_baseline_f1 = 0.4211465842773773
    if best_f1 <= ridge_baseline_f1:
        print("Logistic model did not beat the ridge baseline.")
        print("Skipping final training and submission generation.")
        return

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

    initial_w_final = np.zeros(x_train_final.shape[1])

    # Train the final regularized logistic-regression model.
    weights, loss = reg_logistic_regression(
        y_train_01,
        x_train_final,
        lambda_=lambda_,
        initial_w=initial_w_final,
        max_iters=max_iters,
        gamma=gamma,
    )

    print("Training loss:", loss)

    # Convert continuous scores into {-1, 1} predictions.
    scores = x_test_final @ weights
    predictions = np.where(scores >= best_threshold, 1, -1)

    # Produce the AIcrowd submission.
    output_path = project_directory / "submission_logistic.csv"
    create_csv_submission(test_ids, predictions, output_path)

    print("Submission created:", output_path)


if __name__ == "__main__":
    main()
