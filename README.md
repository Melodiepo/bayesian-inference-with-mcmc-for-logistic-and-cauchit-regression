# Bayesian Inference with MCMC for Logistic and Cauchit Regression

This repository contains the implementation of Bayesian inference using Markov Chain Monte Carlo (MCMC) methods for logistic and cauchit regression models. It includes modules for data simulation, posterior computation, MCMC sampling algorithms, and diagnostics for analyzing convergence and predictive performance. The original R script is also provided for reference.

## Repository Structure

### **1. `data_simulation.py`**
This file contains functions for:
- Simulating the covariate matrix (`X`) and true parameter values (`beta`).
- Generating binary outcomes from logistic and cauchit distributions.

### **2. `posteriors.py`**
This file defines the log-prior and log-posterior functions used in Bayesian inference:
- Gaussian and Unit Information Priors (UIP).
- Posterior distributions for logistic and cauchit regression models.

### **3. `mcmc_models.py`**
This file implements five MCMC sampling algorithms:
1. **Random Walk Metropolis (RWM):** A standard MCMC sampler with Gaussian proposals.
2. **Pre-conditioned RWM (multivariate normal):** Uses covariance scaling for efficient sampling.
3. **Pre-conditioned RWM (multivariate t):** Uses heavy-tailed t-distributed proposals.
4. **Metropolis-Adjusted Langevin Algorithm (MALA):** Incorporates gradient information for efficient proposals.
5. **Component-wise RWM:** Updates each parameter independently.

### **4. `mcmc_diagnostics.py`**
This file contains diagnostic tools to evaluate MCMC performance:
- **Prediction Accuracy:** Calculates predictive performance using posterior means.
- **Brier Score:** Measures probabilistic accuracy of predictions.
- **Gelman-Rubin Statistic:** Evaluates convergence across multiple chains.
- **Autocorrelation Analysis:** Computes and visualizes correlations between MCMC samples.
- **Effective Sample Size (ESS):** Quantifies the number of independent samples.

### **5. `r-script.R`**
The original R script used for the coursework. It includes:
- Data simulation for logistic and cauchit regression models.
- Posterior and prior definitions.
- Implementation of the MCMC algorithms.
- Diagnostic analyses for MCMC results.

### **6. `coursework-report.pdf`**
A PDF report summarizing the coursework and methodology.
