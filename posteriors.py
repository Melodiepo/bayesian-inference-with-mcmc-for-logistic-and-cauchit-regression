import numpy as np
from scipy.stats import norm

def log_prior_gaussian(beta):
    return np.sum(norm.logpdf(beta))

def log_prior_uip(beta, X):
    n = X.shape[0]
    Sigma_inv = np.linalg.inv(X.T @ X) * n
    return -0.5 * beta.T @ Sigma_inv @ beta

def logistic_posterior_gaussian(beta, X, Y):
    p = 1 / (1 + np.exp(-X @ beta))
    log_likelihood = np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    log_prior = log_prior_gaussian(beta)
    return log_likelihood + log_prior

def logistic_posterior_uip(beta, X, Y):
    p = 1 / (1 + np.exp(-X @ beta))
    log_likelihood = np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    log_prior = log_prior_uip(beta, X)
    return log_likelihood + log_prior

def cauchit_posterior_gaussian(beta, X, Y):
    eta = X @ beta
    p = 0.5 + (1 / np.pi) * np.arctan(eta)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    log_likelihood = np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    log_prior = log_prior_gaussian(beta)
    return log_likelihood + log_prior

def cauchit_posterior_uip(beta, X, Y):
    eta = X @ beta
    p = 0.5 + (1 / np.pi) * np.arctan(eta)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    log_likelihood = np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    log_prior = log_prior_uip(beta, X)
    return log_likelihood + log_prior
