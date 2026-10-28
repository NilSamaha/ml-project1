from pathlib import Path

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import ridge_regression

def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0

def main():
    # Locate the data folder relative to this file.
    project_directory = Path(__file__).resolve().parent
    data_directory = project_directory / "data"

    # Load the dataset.
    x_train, x_test, y_train, _, test_ids = load_csv_data(data_directory)

    print("Training shape:", x_train.shape)
    print("Test shape:", x_test.shape)

    # Replace missing values with training-column medians.
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)

    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_test = np.where(np.isnan(x_test), medians, x_test)

    # Standardize using statistics from the training set.
    means = np.mean(x_train, axis=0)
    standard_deviations = np.std(x_train, axis=0)
    standard_deviations[standard_deviations == 0] = 1

    x_train = (x_train - means) / standard_deviations
    x_test = (x_test - means) / standard_deviations

    # Add a bias column.
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))

    # Split the training set into a training and validation set.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_train))
    cut = int(0.8 * len(idx))
    tr, va = idx[:cut], idx[cut:]

    w_val, _ = ridge_regression(y_train[tr], x_train[tr], lambda_=0.001)
    scores_val = x_train[va] @ w_val
    pred_val = np.where(scores_val >= 0, 1, -1)

    print("Val accuracy:", np.mean(pred_val == y_train[va]))
    print("Val F1:", f1_score(y_train[va], pred_val))

    # Train a basic ridge-regression model.
    weights, loss = ridge_regression(y_train, x_train, lambda_=0.001)

    print("Training loss:", loss)

    # Convert continuous scores into {-1, 1} predictions.
    scores = x_test @ weights
    predictions = np.where(scores >= 0, 1, -1)

    # Produce the AIcrowd submission.
    output_path = project_directory / "submission.csv"
    create_csv_submission(test_ids, predictions, output_path)

    print("Submission created:", output_path)


if __name__ == "__main__":
    main()