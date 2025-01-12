import numpy as np

def simulate_data(n, d):
    np.random.seed(692)
    X = np.zeros((n, d))
    X[:, 0] = 1  # Intercept
    z = np.random.randn(n, d - 1)
    X[:, 1] = z[:, 0]
    X[:, 2] = z[:, 0] + 0.2 * z[:, 1]
    X[:, 3] = 0.5 * z[:, 2]
    X[:, 4] = z[:, 3]
    X[:, 5] = 2 * z[:, 4] + 20 * z[:, 5]
    X[:, 6] = z[:, 5]
    X[:, 7] = 0.5 * (z[:, 6] + z[:, 3] + z[:, 7] + z[:, 0])
    X[:, 8] = z[:, 7] + 10 * z[:, 3]
    X[:, 9] = z[:, 4] + 0.5 * z[:, 8]

    beta = np.linspace(-2, 2, d)
    return X, beta


from scipy.stats import cauchy

def generate_logistic_y(X, beta, n_samples=150):
    eps = np.random.logistic(size=n_samples)
    Y_star = X @ beta + eps
    return (Y_star > 0).astype(int)

def generate_cauchy_y(X, beta, n_samples=150):
    eps = cauchy.rvs(size=n_samples)
    Y_star = X @ beta + eps
    return (Y_star > 0).astype(int)
