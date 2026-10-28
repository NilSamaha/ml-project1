import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Perform linear regression using gradient descent.

    Args:
        y: Target values with shape (n,).
        tx: Feature matrix with shape (n, d).
        initial_w: Initial weight vector with shape (d,).
        max_iters: Number of gradient-descent iterations.
        gamma: Learning rate.

    Returns:
        A tuple containing the final weight vector and the mean squared error.
    """
    w = initial_w
    n = np.shape(y)[0]
    for i in range(max_iters):
        e = tx @ w - y
        grad = 1 / n * np.transpose(tx) @ (e)
        w = w - gamma * grad
    e = tx @ w - y
    loss = np.sum(e**2) / (2 * n)
    return (w, loss)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Perform linear regression using stochastic gradient descent.

    Args:
        y: Target values with shape (n,).
        tx: Feature matrix with shape (n, d).
        initial_w: Initial weight vector with shape (d,).
        max_iters: Number of stochastic gradient-descent iterations.
        gamma: Learning rate.

    Returns:
        A tuple containing the final weight vector and the mean squared error.
    """
    rng = np.random.default_rng()
    w = initial_w
    n = np.shape(y)[0]
    for i in range(max_iters):
        index = rng.integers(0, len(y))
        x_i = tx[index]
        y_i = y[index]
        e = x_i @ w - y_i
        grad = e * x_i
        w = w - gamma * grad
    e = tx @ w - y
    loss = np.sum(e**2) / (2 * n)
    return (w, loss)


def least_squares(y, tx):
    """Compute the least-squares solution using the normal equations.

    Args:
        y: Target values with shape (n,).
        tx: Feature matrix with shape (n, d).

    Returns:
        A tuple containing the optimal weight vector and the mean squared error.
    """
    n = np.shape(y)[0]
    rh = np.transpose(tx) @ y
    lh = np.transpose(tx) @ tx
    w = np.linalg.solve(lh, rh)
    e = tx @ w - y
    return (w, np.sum(e**2) / (2 * n))


def ridge_regression(y, tx, lambda_):
    """Compute the ridge-regression solution using the normal equations.

    Args:
        y: Target values with shape (n,).
        tx: Feature matrix with shape (n, d).
        lambda_: Regularization strength.

    Returns:
        A tuple containing the regularized weight vector and its mean squared error.
    """
    n = np.shape(y)[0]
    dim = np.shape(tx)[1]
    lh = np.transpose(tx) @ tx + 2 * n * lambda_ * np.identity(dim)
    rh = np.transpose(tx) @ y
    w = np.linalg.solve(lh, rh)
    loss = np.sum((y - tx @ w) ** 2) / (2 * n)
    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform binary logistic regression using gradient descent.

    Args:
        y: Binary target values with shape (n,).
        tx: Feature matrix with shape (n, d).
        initial_w: Initial weight vector with shape (d,).
        max_iters: Number of gradient-descent iterations.
        gamma: Learning rate.

    Returns:
        A tuple containing the final weight vector and the logistic loss.
    """
    n = np.shape(y)[0]
    w = initial_w
    for i in range(max_iters):
        z = tx @ w
        sigma = 1 / (1 + np.exp(-z))
        grad = np.transpose(tx) @ (sigma - y) / n
        w = w - gamma * grad
    z = tx @ w
    loss = np.sum(np.logaddexp(0, z) - y * z) / n
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Perform regularized binary logistic regression using gradient descent.

    Args:
        y: Binary target values with shape (n,).
        tx: Feature matrix with shape (n, d).
        lambda_: Regularization strength.
        initial_w: Initial weight vector with shape (d,).
        max_iters: Number of gradient-descent iterations.
        gamma: Learning rate.

    Returns:
        A tuple containing the regularized weight vector and its logistic loss.
    """
    n = np.shape(y)[0]
    w = initial_w
    for i in range(max_iters):
        z = tx @ w
        sigma = 1 / (1 + np.exp(-z))
        grad = np.transpose(tx) @ (sigma - y) / n + 2 * lambda_ * w
        w = w - gamma * grad
    z = tx @ w
    loss = np.sum(np.logaddexp(0, z) - y * z) / n
    return (w, loss)
