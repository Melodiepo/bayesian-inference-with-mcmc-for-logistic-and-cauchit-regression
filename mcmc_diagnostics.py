import numpy as np
from scipy.stats import entropy

# 1. Prediction Accuracy
def prediction_accuracy(beta_estimates, X, Y):
    """
    Computes prediction accuracy by comparing predicted outcomes to true outcomes.
    """
    p_pred = 1 / (1 + np.exp(-X @ beta_estimates))  # Predicted probabilities
    predictions = (p_pred > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)
    return accuracy

# 1.1 Brier Score
def brier_score(Y_true, Y_pred):
    """
    Computes the Brier score as a measure of predictive accuracy.
    """
    return np.mean((Y_true - Y_pred) ** 2)

# 2. Gelman-Rubin Diagnostic
def gelman_rubin(chains):
    """
    Computes the Gelman-Rubin diagnostic for MCMC convergence.
    """
    num_chains, num_samples, num_params = chains.shape
    chain_means = chains.mean(axis=1)  # Mean of each chain
    overall_mean = chain_means.mean(axis=0)  # Overall mean across chains

    B = num_samples * np.sum((chain_means - overall_mean) ** 2, axis=0) / (num_chains - 1)
    W = np.sum((chains - chain_means[:, np.newaxis, :]) ** 2, axis=(0, 1)) / (num_chains * (num_samples - 1))
    R_hat = np.sqrt((B / W) + (num_samples - 1) / num_samples)
    return R_hat

# 3. Autocorrelation Analysis
def compute_autocorrelation(samples, max_lag=50):
    """
    Computes autocorrelation for MCMC samples up to a maximum lag.
    """
    n, d = samples.shape
    autocorr = np.zeros((max_lag, d))
    for lag in range(1, max_lag + 1):
        autocorr_lag = np.mean((samples[:-lag] - samples.mean(axis=0)) * (samples[lag:] - samples.mean(axis=0)), axis=0)
        autocorr_lag /= np.var(samples, axis=0)
        autocorr[lag - 1] = autocorr_lag
    return autocorr

def plot_autocorrelation(autocorr, parameter_names):
    """
    Plots autocorrelation for each parameter.
    """
    import matplotlib.pyplot as plt

    lags = np.arange(1, autocorr.shape[0] + 1)
    for i, name in enumerate(parameter_names):
        plt.plot(lags, autocorr[:, i], label=name)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation for MCMC Parameters")
    plt.legend()
    plt.show()

# 4. Effective Sample Size (ESS)
def effective_sample_size(samples):
    """
    Computes the effective sample size (ESS) for MCMC samples.
    """
    n, d = samples.shape
    autocorr_sum = np.zeros(d)
    for lag in range(1, n):
        autocorr = np.mean((samples[:-lag] - samples.mean(axis=0)) * (samples[lag:] - samples.mean(axis=0)), axis=0)
        autocorr /= np.var(samples, axis=0)
        if np.all(autocorr < 0):
            break
        autocorr_sum += 2 * autocorr
    ess = n / (1 + autocorr_sum)
    return ess

# Helper: Posterior Mean Estimation
def posterior_means(samples):
    """
    Computes the posterior mean of the samples for each parameter.
    """
    return np.mean(samples, axis=0)

# Helper: Error Matrix
def compute_error_matrix(samples, true_beta):
    """
    Computes the error between posterior means and true beta coefficients.
    """
    posterior_mean = posterior_means(samples)
    return posterior_mean - true_beta
