import numpy as np
from scipy.stats import multivariate_normal, t

# 1. Random Walk Metropolis (RWM)
def random_walk_metropolis(log_posterior, n_iters, step_size, initial, X, Y):
    d = len(initial)
    samples = np.zeros((n_iters, d))
    current = initial
    current_log_post = log_posterior(current, X, Y)
    accepted = 0

    for i in range(n_iters):
        proposal = current + step_size * np.random.randn(d)
        proposal_log_post = log_posterior(proposal, X, Y)

        if np.log(np.random.rand()) < proposal_log_post - current_log_post:
            current = proposal
            current_log_post = proposal_log_post
            accepted += 1

        samples[i] = current
    acceptance_rate = accepted / n_iters
    return samples, acceptance_rate

# 2. Pre-conditioned RWM with multivariate normal proposals
def preconditioned_rwm_mnorm(log_posterior, n_iters, cov_matrix, initial, X, Y):
    d = len(initial)
    samples = np.zeros((n_iters, d))
    current = initial
    current_log_post = log_posterior(current, X, Y)
    accepted = 0

    for i in range(n_iters):
        proposal = current + multivariate_normal.rvs(mean=np.zeros(d), cov=cov_matrix)
        proposal_log_post = log_posterior(proposal, X, Y)

        if np.log(np.random.rand()) < proposal_log_post - current_log_post:
            current = proposal
            current_log_post = proposal_log_post
            accepted += 1

        samples[i] = current
    acceptance_rate = accepted / n_iters
    return samples, acceptance_rate

# 3. Pre-conditioned RWM with multivariate t proposals
def preconditioned_rwm_mt(log_posterior, n_iters, cov_matrix, df, initial, X, Y):
    d = len(initial)
    samples = np.zeros((n_iters, d))
    current = initial
    current_log_post = log_posterior(current, X, Y)
    accepted = 0

    for i in range(n_iters):
        proposal = current + t.rvs(df=df, size=d) @ np.linalg.cholesky(cov_matrix).T
        proposal_log_post = log_posterior(proposal, X, Y)

        if np.log(np.random.rand()) < proposal_log_post - current_log_post:
            current = proposal
            current_log_post = proposal_log_post
            accepted += 1

        samples[i] = current
    acceptance_rate = accepted / n_iters
    return samples, acceptance_rate

# 4. Metropolis-Adjusted Langevin Algorithm (MALA)
def metropolis_adjusted_langevin(log_posterior, grad_log_posterior, n_iters, step_size, initial, X, Y):
    d = len(initial)
    samples = np.zeros((n_iters, d))
    current = initial
    current_log_post = log_posterior(current, X, Y)
    grad_current = grad_log_posterior(current, X, Y)
    accepted = 0

    for i in range(n_iters):
        proposal = current + 0.5 * step_size * grad_current + np.sqrt(step_size) * np.random.randn(d)
        proposal_log_post = log_posterior(proposal, X, Y)
        grad_proposal = grad_log_posterior(proposal, X, Y)

        log_q_current = -0.5 * np.sum((current - (proposal + 0.5 * step_size * grad_proposal)) ** 2) / step_size
        log_q_proposal = -0.5 * np.sum((proposal - (current + 0.5 * step_size * grad_current)) ** 2) / step_size

        if np.log(np.random.rand()) < (proposal_log_post - current_log_post + log_q_current - log_q_proposal):
            current = proposal
            current_log_post = proposal_log_post
            grad_current = grad_proposal
            accepted += 1

        samples[i] = current
    acceptance_rate = accepted / n_iters
    return samples, acceptance_rate

# 5. Component-wise RWM
def component_wise_rwm(log_posterior, n_iters, step_sizes, initial, X, Y):
    d = len(initial)
    samples = np.zeros((n_iters, d))
    current = initial
    current_log_post = log_posterior(current, X, Y)
    accepted = np.zeros(d)

    for i in range(n_iters):
        for j in range(d):
            proposal = current.copy()
            proposal[j] += step_sizes[j] * np.random.randn()
            proposal_log_post = log_posterior(proposal, X, Y)

            if np.log(np.random.rand()) < proposal_log_post - current_log_post:
                current[j] = proposal[j]
                current_log_post = proposal_log_post
                accepted[j] += 1

        samples[i] = current
    acceptance_rate = accepted / n_iters
    return samples, acceptance_rate
