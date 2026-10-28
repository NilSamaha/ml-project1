# EPFL CS-433 Machine Learning - Project 1

This repository contains our solution for the EPFL CS-433 Machine Learning
Project 1: binary classification of cardiovascular disease (MICHD) from the
Behavioral Risk Factor Surveillance System data. The final system uses ridge
regression with preprocessing designed for a large, sparse medical survey
dataset.

## Repository structure

- `implementations.py`: implementations of the six required regression and
  classification methods.
- `helpers.py`: data loading, CSV submission, preprocessing, and evaluation
  utilities.
- `run.py`: complete reproducible pipeline for the final ridge-regression
  submission.
- `run_logistic.py`: regularized logistic-regression comparison experiment.

## Requirements

The project uses only Python standard-library modules and NumPy, as required by
the project rules. It was developed with Python 3.11.

Install NumPy if it is not already available:

```bash
python3 -m pip install numpy
```

## Data

Download the Project 1 dataset from the course or AIcrowd page. Place the three
CSV files in a `data` directory at the repository root:

```text
data/
├── x_train.csv
├── x_test.csv
└── y_train.csv
```

The data files are intentionally excluded from Git because of their size.

## Reproducing the final submission

From the repository root, run:

```bash
python3 run.py
```

The script performs model selection, trains the selected model on all labeled
training examples, and creates `submission.csv` in the repository root. CSV
generation is deterministic because the validation split uses NumPy random seed
`0`.

The full CSV files are large, so data loading and training can take several
minutes and require several gigabytes of memory.

## Method

### Validation

We use a reproducible 80/20 train-validation split. Preprocessing parameters are
fitted only on the training portion and then applied unchanged to validation
data, preventing data leakage. After model selection, preprocessing is refitted
on all labeled training data and applied to the hidden test data.

### Preprocessing

1. Replace missing values using medians learned from the training data.
2. Standardize every original feature using training means and standard
   deviations.
3. Add binary indicators for columns containing missing values. These preserve
   information about whether a value was originally absent before imputation.
4. Add a constant bias feature.

### Final model

The final classifier is ridge regression. We compare regularization strengths

```text
1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
```

on the validation set. For every value, we tune the decision threshold over 200
validation-score quantiles and select the configuration with the highest F1
score. The selected configuration uses `lambda = 1e-5` and a validation
threshold of approximately `-0.5631`.

## Results

| Experiment | Validation accuracy | Validation F1 |
| --- | ---: | ---: |
| Initial ridge, threshold 0 | 0.9116 | 0.0143 |
| Ridge, tuned threshold | 0.8566 | 0.4127 |
| Leakage-free preprocessing | 0.8567 | 0.4129 |
| Ridge, tuned regularization | 0.8556 | 0.4144 |
| Ridge with missing indicators | **0.8715** | **0.4211** |
| Regularized logistic regression, 100 iterations | 0.8536 | 0.4004 |
| Regularized logistic regression, 300 iterations | 0.8691 | 0.4103 |

The final ridge submission achieved an AIcrowd accuracy of **0.874** and an F1
score of **0.429**.

The initial accuracy was misleading because the positive class represents only
about 8.8% of the training examples. Threshold tuning substantially improved F1
by allowing the model to identify more positive cases. Missing-value indicators
provided a further measurable improvement. Regularized logistic regression
improved with additional gradient-descent iterations but remained below the
ridge baseline.

## Logistic-regression comparison

The comparison experiment can be reproduced with:

```bash
python3 run_logistic.py
```

It uses `lambda = 1e-5`, learning rate `0.1`, and 300 iterations. The script
stops after validation when it does not beat the final ridge validation F1,
avoiding unnecessary full-data training.

## Required method implementations

`implementations.py` provides:

- Mean squared error regression using gradient descent
- Mean squared error regression using stochastic gradient descent
- Least-squares regression using the normal equations
- Ridge regression using the normal equations
- Logistic regression using gradient descent
- Regularized logistic regression using gradient descent

All functions return the final weight vector and the corresponding unpenalized
loss, following the required project signatures.
