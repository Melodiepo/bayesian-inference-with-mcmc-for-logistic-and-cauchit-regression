library(mvtnorm)
library(coda)
library(ggplot2)
library(reshape2)
library(dplyr)
library(MASS)

                         # Section 1: Data Simulation 
set.seed(0692) 

n <- 150 # number of observations
d <- 10 # number of beta parameters

# create matrix to populate with covariates
X <- matrix(nrow = n, ncol = d)
X[,1] <- rep(1, n) # first column is an intercept

# create base uncorrelated random numbers to turn into x_i's
z <- matrix(rnorm(n*(d-1)), nrow = n, ncol = d-1)

# create x_i's (ith column of matrix corresponds to variable x_i)
X[,2] <- z[,1]
X[,3] <- z[,1] + 0.2*z[,2]
X[,4] <- 0.5*z[,3]
X[,5] <- z[,4]
X[,6] <- 2*z[,5] + 20*z[,6]
X[,7] <- z[,6]
X[,8] <- 0.5 * (z[,7] + z[,4] + z[,8] + z[,1])
X[,9] <- z[,8] + 10*z[,4]
X[,10] <- z[,5] + 0.5*z[,9]

# create true beta values
beta <- seq(-2,2, length = 10)

# Section 1.1: Generating Binary Outcomes from Logistic Distribution 
# Define the PDF of the logistic distribution
f_L <- function(x) exp(-x) / (1 + exp(-x))^2

# Define the PDF of the Cauchy distribution
f_C <- function(x) 1 / (pi * (1 + x^2))

# Define the ratio of logistic to Cauchy densities
ratio_LC <- function(x) {
  f_L(x) / f_C(x)
}

# Find the optimized M
M_calc <- optimize(f = ratio_LC, interval = c(-10,10), maximum = TRUE)
M <- M_calc$objective
M

n_samples <- n  # the number of samples we want to draw
accepted_samples <- numeric(n_samples)  # to store accepted samples
num_accepted <- 0  # to keep track of the number of accepted samples
num_proposed <- 0  # to keep track of the number of proposed samples

while (num_accepted < n_samples) {
    x_proposed <- rt(1, df = 1)  # draw a sample from the Cauchy distribution
    fl_value <- f_L(x_proposed)  # calculate the logistic PDF value

    if (is.finite(fl_value)) {
        u <- runif(1, 0, M * fl_value)  # draw a sample from the uniform distribution

        if (u <= fl_value) {
            num_accepted <- num_accepted + 1
            accepted_samples[num_accepted] <- x_proposed
        }
    }
    num_proposed <- num_proposed + 1
}

# Estimate efficiency
efficiency <- num_accepted / num_proposed
efficiency

# Now use these samples to create Y
epsilon_L <- accepted_samples  # These are our accepted epsilon_i samples
Y_star_L <- X %*% beta + epsilon_L  # Calculate the latent variable Y_star
Y_L <- ifelse(Y_star_L > 0, 1, 0)  # Convert the latent variable to binary outcomes Y

# Section 1.2: Generating Binary Outcomes from Cauchy Distribution 
# Generate the random errors from the Cauchy distribution, i.e. t-distribution with df = 1
epsilon_C <- rt(n, df = 1)

# Calculate the latent variable using the matrix product of X and beta, plus the error term (formula given in the instrutioon)
Y_star_C <- X %*% beta + epsilon_C

# Convert the latent variable to a binary outcome
Y_C <- ifelse(Y_star_C > 0, 1, 0)


# Section 2: Model Sepcification: Defining Prior and Posterior
# Gaussian log-prior function
log_prior_gaussian <- function(beta) {
 sum(dnorm(beta, log = T))
}

# Unit Information Prior(UIP) log-prior function 
log_prior_uip <- function(beta, X) {
  n <- nrow(X)  # Calculate 'n' as the number of rows in 'X'
  Sigma_inv <- solve(t(X) %*% X) * n
  prior_value <- -0.5 * t(beta) %*% Sigma_inv %*% beta
  return(prior_value)
}


# Logistic-Gaussian Posterior
logistic_posterior_Gaussian <- function(beta, X, Y) {
  p <- 1 / (1 + exp(-X %*% beta))
  log_likelihood <- sum(dbinom(Y, size = 1, prob = p, log = TRUE))
  log_prior <- log_prior_gaussian(beta)
  return(log_likelihood + log_prior)
}


# Logistic-UIP Posterior
logistic_posterior_UIP <- function(beta, X, Y) {
  # Ensure beta is treated as a column vector for matrix operations
  if (is.null(dim(beta))) {
    beta <- matrix(beta, nrow = length(beta), ncol = 1)
  }

  # Calculate logistic model probabilities
  p <- 1 / (1 + exp(-X %*% beta))
  log_likelihood <- sum(Y * log(p) + (1 - Y) * log(1 - p))
  
  # Calculate log prior using Sigma_inv
  log_prior <- log_prior_uip(beta, X)
  
  return(log_likelihood + log_prior)
}

# Cauchit-Gaussian Posterior
cauchit_posterior_Gaussian <- function(beta, X, Y) {
  eta <- X %*% beta
  p <- 0.5 + (1 / pi) * atan(eta) 
  p <- pmax(pmin(p, 1 - 1e-10), 1e-10)  # Avoid log(0)
  log_likelihood <- sum(dbinom(Y, size = 1, prob = p, log = TRUE))
  
  log_prior <- sum(dnorm(beta, log = T))
  
  return(log_likelihood + log_prior)
}

# Cauchit-UIP Posterior
cauchit_posterior_UIP <- function(beta, X, Y) {
  eta <- X %*% beta
  p <- 0.5 + (1 / pi) * atan(eta) 
  p <- pmax(pmin(p, 1 - 1e-10), 1e-10)  # Avoid log(0)
  log_likelihood <- sum(dbinom(Y, size = 1, prob = p, log = TRUE))
  
  log_prior <- log_prior_uip(beta,X)
  
  return(log_likelihood + log_prior)
}





# Section 3: MCMC Algorithm Implementation and Application

# This section is dedicated to the implementation and fitting of five distinct 
# MCMC algorithms to logistic and Cauchit regression models. Each algorithm is
# applied to both Gaussian and UIP Prior cases to assess model 
# performance. The five algorithms are:

# 1. Random Walk Metropolis (RWM)
# 2. Pre-conditioned RWM using multivariate normal proposals
# 3. Pre-conditioned RWM using multivariate t proposals
# 4. Metropolis-adjusted Langevin Algorithm (MALA)
# 5. Component-wise RWM

# For each algorithm, the process consists of defining a fitting function, 
# executing the MCMC sampling, and calculating the acceptance rate. After 
# fitting to the Logistic case, the same algorithms are applied to the Cauchit 
# regression model. Results are visualized through trace plots.


# Before going through the five MCMC algorithms specific to each regression
# We want to define some common parameters used in MCMC to reduce redundancy
nits <- 20000
x_curr <- rep(0, ncol(X))

                              #Logistic Case
# MODEL: Random Walk Metropolis (RWM)
                             # Logistic-Gaussian
# Multidimensional RWM algorithm for logistic regression
RWM <- function(log_posterior, nits, h, x_curr, X, Y) {
  d <- length(x_curr)
  accepted <- 0
  x_store <- matrix(nrow = nits, ncol = d)
  
  # Initialize current log-posterior
  logpi_curr <- log_posterior(x_curr, X, Y)
  if (is.nan(logpi_curr)) logpi_curr <- -Inf  # Handle NaN

  for (i in 1:nits) {
    # propose a candidate move
    x_prop <- x_curr + h * rnorm(d)
    logpi_prop <- log_posterior(x_prop, X, Y)
    
    # Check for numerical issues
    if (is.nan(logpi_prop)) logpi_prop <- -Inf  # Handle NaN

    # accept-reject step
    loga <- logpi_prop - logpi_curr
    u <- runif(1)
    # compare u < exp(loga) to avoid numerical issues
    if (!is.nan(loga) && u < exp(loga)) {
      x_curr <- x_prop
      logpi_curr <- logpi_prop
      accepted <- accepted + 1
    }
    x_store[i,] <- x_curr
  }
  
  return(list(x_store = x_store, a_rate = accepted / nits))
}

# Run the RWM algorithm for logistic regression with Gaussian
mc_logGaussian_rwm <- RWM(logistic_posterior_Gaussian, nits, h = 0.08, x_curr, X, Y_L)

# Output the acceptance rate
cat("The acceptance rate is: ", mc_logGaussian_rwm$a_rate)

                            # Logistic-UIP
# Run the RWM algorithm for logistic regression with Gaussian
mc_logUIP_rwm <- RWM(logistic_posterior_UIP, nits, h = 0.05, x_curr, X,Y_L)

# Output the acceptance rate
cat("The acceptance rate is: ", mc_logUIP_rwm$a_rate)

                            # Trace plot 
# Set up colors for the plots
colors <- rainbow(ncol(mc_logGaussian_rwm$x_store))

mc_trace_plot <- function(mc_data, colors, model_name = "Model") {
  # Loop through each parameter in the MCMC data
  for (i in 1:ncol(mc_data$x_store)) {
    # Trace plot for the i-th beta parameter with the i-th color
    plot(mc_data$x_store[, i], type = 'l',
         main = paste("Trace for Beta", i, "in", model_name),
         ylab = paste("Beta", i), xlab = "Iteration",
         col = colors[i])
    
    # Convert the color to RGB with transparency
    col_transparent <- adjustcolor(colors[i], alpha.f = 0.3)
  }
}

# Call the function for the Logistic-Gaussian/UIP RWM model
mc_trace_plot(mc_logGaussian_rwm, colors, "Logistic-Gaussian RWM")
mc_trace_plot(mc_logUIP_rwm, colors, "Logistic-Gaussian RWM")




# MODEL: Pre-conditioned RWM using multivariate normal proposals
                         #Logistic-Gaussian
# RWM algorithm function with pre-conditioning
RWM_pre_mnorm <- function(log_posterior, X, Y, nits, cov_matrix, x_curr) {
  d <- length(x_curr)
  accepted <- 0
  x_store <- matrix(nrow = nits, ncol = d)
  
  for (i in 1:nits) {
    # Generate a proposal beta vector by adding a random multivariate normal vector
    # The covariance of the random vector is the provided cov_matrix
    x_prop <- x_curr + mvrnorm(1, mu = rep(0, d), Sigma = cov_matrix)
    
    # Calculate the log of the acceptance ratio
    loga <- log_posterior(x_prop, X, Y) - log_posterior(x_curr, X, Y)
    
    # Accept or reject the proposal based on the acceptance ratio
    if (is.finite(loga) && runif(1) < exp(loga)) {
      x_curr <- x_prop
      accepted <- accepted + 1
    }
    # Store the current or last accepted beta vector in the chain
    x_store[i, ] <- x_curr
  }
  
  return(list(x_store = x_store, a_rate = accepted / nits))
}

# Define parameters for the initial run
nits_initial <- 20000  # Total number of iterations including burn-in
burn_in <- 5000        # Number of iterations to discard as burn-in
h_initial <- 0.003      # Initial proposal width, used to scale the identity covariance matrix
x_curr_initial <- rep(0, ncol(X))

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run <- RWM_pre_mnorm(logistic_posterior_Gaussian, X, Y_L, nits_initial, diag(rep(h_initial, ncol(X))), x_curr_initial)
# Tune the step-size to achieve a 30-40% acceptance rate
initial_run$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
nits_final <- 20000
h_final <- 0.45  

# Run the RWM algorithm with the estimated covariance matrix
mc_logGaussian_pre_norm <- RWM_pre_mnorm(logistic_posterior_Gaussian, X, Y_L, nits_final, cov_matrix_est * h_final, x_curr_initial)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logGaussian_pre_norm$a_rate)

                            #Logistic-UIP
# RWM algorithm function with pre-conditioning
RWM_pre_mnorm

# Define parameters for the initial run
nits_initial <- 20000  
burn_in <- 5000        
h_initial <- 0.0015  
x_curr_initial <- rep(0, ncol(X))

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run <- RWM_pre_mnorm(logistic_posterior_UIP, X, Y_L, nits_initial, diag(rep(h_initial, ncol(X))), x_curr_initial)
# Tune the step-size to achieve a 30-40% acceptance rate
initial_run$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
nits_final <- 20000
h_final <- 0.45  

# Run the RWM algorithm with the estimated covariance matrix
mc_logUIP_pre_norm <- RWM_pre_mnorm(logistic_posterior_UIP, X, Y_L, nits_final, cov_matrix_est * h_final, x_curr_initial)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logUIP_pre_norm$a_rate)

# Trace plot for Logistic-Gaussian and Logistic-UIP 
mc_trace_plot(mc_logGaussian_pre_norm, colors, "Pre Logistic-Gaussian mnorm")
mc_trace_plot(mc_logUIP_pre_norm, colors, "Pre Logistic-UIP mnorm")



# MODEL: Pre-conditioned RWM using multivariate t proposals
                            #Logistic-Gaussian
# RWM algorithm function with pre-conditioning
RWM_pre_mt <- function(log_posterior, X, Y, nits, cov_matrix, x_curr, df_t) {
  d <- length(x_curr)
  accepted <- 0
  x_store <- matrix(nrow = nits, ncol = d)
  
  for (i in 1:nits) {
    # Generate a proposal beta vector by adding a random multivariate normal vector
    # The covariance of the random vector is the provided cov_matrix
    x_prop <- x_curr + as.vector(rmvt(n = 1, sigma = cov_matrix, df = df_t))
    
    # Calculate the log of the acceptance ratio
    loga <- log_posterior(x_prop, X, Y) - log_posterior(x_curr, X, Y)
    
    # Accept or reject the proposal based on the acceptance ratio
    if (is.finite(loga) && runif(1) < exp(loga)) {
      x_curr <- x_prop
      accepted <- accepted + 1
    }
    # Store the current or last accepted beta vector in the chain
    x_store[i, ] <- x_curr
  }
  
  return(list(x_store = x_store, a_rate = accepted / nits))
}

h_initial <- 0.001 # Initial proposal width, used to scale the identity covariance matrix
df_t <- 1

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run_t <- RWM_pre_mt(logistic_posterior_Gaussian, X, Y_L, nits_initial, 
                            diag(rep(h_initial, ncol(X))), x_curr_initial, df_t)

# Tune the step-size to achieve a 30-40% acceptance rate
initial_run_t$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run_t$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
h_final <- 0.2 

# Run the RWM algorithm with the estimated covariance matrix
mc_logGaussian_pre_t <- RWM_pre_mt(logistic_posterior_Gaussian, X, Y_L, nits_final, 
                                   cov_matrix_est*h_final, x_curr_initial, df_t)


# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logGaussian_pre_t$a_rate)

                               #Logistic-UIP
# RWM algorithm function with pre-conditioning & multivariate t proposal
RWM_pre_mt
# Define parameters for the initial run
h_initial <- 0.001      # Initial proposal width, used to scale the identity covariance matrix
df_t <- 1

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run_t <- RWM_pre_mt(logistic_posterior_UIP, X, Y_L, nits_initial, diag(rep(h_initial, ncol(X))), x_curr_initial, df_t)

# Tune the step-size to achieve a 30-40% acceptance rate
initial_run_t$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run_t$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
nits_final <- 20000
h_final <- 0.35

# Run the RWM algorithm with the estimated covariance matrix
mc_logUIP_pre_t <- RWM_pre_mt(logistic_posterior_UIP, X, Y_L, nits_final, cov_matrix_est*h_final, x_curr_initial, df_t)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logUIP_pre_t$a_rate)

# Trace plot for Logistic-UIP preconditioned_norm
mc_trace_plot(mc_logGaussian_pre_t, colors, "Pre Logistic-Gaussian mt")
mc_trace_plot(mc_logUIP_pre_t, colors, "Pre Logistic-UIP mt")



# MODEL: Metropolis-adjusted Langevin algorithm (MALA)
                              # Logistic-Gaussian
MALA <- function(logistic_posterior, dlogpi, nits, h, x_curr, X, Y) {
  d <- length(x_curr)
  accepted <- 0
  x_store <- matrix(nrow = nits, ncol = d)
  
  # Initialize current log-posterior and its gradient
  logpi_curr <- logistic_posterior(x_curr, X, Y)
  if (is.nan(logpi_curr)) logpi_curr <- -Inf
  dlogpi_curr <- dlogpi(x_curr, X, Y)
  
  for (i in 1:nits) {
    # Propose a candidate move
    x_prop <- x_curr + h * dlogpi_curr / 2 + sqrt(h) * rnorm(d)
    logpi_prop <- logistic_posterior(x_prop, X, Y)
    dlogpi_prop <- dlogpi(x_prop, X, Y)
    
    # Convert matrices to vectors explicitly
    x_curr <- as.vector(x_curr)
    x_prop <- as.vector(x_prop)
    dlogpi_curr <- as.vector(dlogpi_curr)
    dlogpi_prop <- as.vector(dlogpi_prop)

    # Calculate transition probabilities
    logq_cgivenp <- dmvnorm(x_curr, mean = x_prop + h * dlogpi_prop / 2, sigma = h * diag(rep(1, d)), log = TRUE)
    logq_pgivenc <- dmvnorm(x_prop, mean = x_curr + h * dlogpi_curr / 2, sigma = h * diag(rep(1, d)), log = TRUE)
    
    # Accept-reject step
    loga <- logpi_prop - logpi_curr + logq_cgivenp - logq_pgivenc
    u <- runif(1)
    # compare u < exp(loga) to avoid numerical issues
    if (!is.nan(loga) && u < exp(loga)) {
      x_curr <- x_prop
      logpi_curr <- logpi_prop
      dlogpi_curr <- dlogpi_prop
      accepted <- accepted + 1
    }
    x_store[i, ] <- x_curr
  }
  
  return(list(x_store = x_store, a_rate = accepted / nits))
}

dlogpi_logGaussian <- function(beta, X, Y) {
  z <- X %*% beta
  p <- 1 / (1 + exp(-z))
  grad_log_likelihood <- t(X) %*% (Y - p)
  grad_log_prior <- -beta
  return(grad_log_likelihood + grad_log_prior)
}

mc_logGaussian_mala <- MALA(logistic_posterior_Gaussian, dlogpi_logGaussian, 
                            nits, h = 0.001, x_curr, X, Y_L)
                        # Performing burn-in
burn_in <- 5000  # Define the number of burn-in iterations

# Create new MCMC objects that exclude the burn-in period
mc_logGaussian_mala_burnin <- mc_logGaussian_mala
mc_logGaussian_mala_burnin$x_store <- mc_logGaussian_mala$x_store[(burn_in + 1):nrow(mc_logGaussian_mala$x_store), ]


# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logGaussian_mala_burnin$a_rate)

                                # Logistic-UIP
dlogpi_logUIP <- function(beta, X, Y) {
  n <- nrow(X)
  Sigma_inv <- solve(t(X) %*% X) * n
  
  p <- 1 / (1 + exp(-X %*% beta))
  grad_log_likelihood <- t(X) %*% (Y - p)
  
  grad_log_prior <- -Sigma_inv %*% beta
  
  return(grad_log_likelihood + grad_log_prior)
}

mc_logUIP_mala <- MALA(logistic_posterior_UIP, dlogpi_logUIP, nits, h = 0.001, x_curr, X, Y_L)
                           # Performing burn-in
burn_in <- 5000  

# Create new MCMC objects that exclude the burn-in period
mc_logUIP_mala_burnin <- mc_logUIP_mala 
mc_logUIP_mala_burnin$x_store <- mc_logUIP_mala$x_store[(burn_in + 1):nrow(mc_logUIP_mala$x_store), ]

chains <- mcmc.list(mcmc(mc_logUIP_mala_burnin$x_store))

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logUIP_mala_burnin$a_rate)

                # Trace plot for Logistic-Gaussian/UIP MALA
mc_trace_plot(mc_logGaussian_mala_burnin, colors, "Post Burn-In Logistic-Gaussian MALA")
mc_trace_plot(mc_logUIP_mala_burnin, colors, "Post Burn-IN Logistic-UIP MALA")


# MODEL: Component-wise RWM
                             # Logistic-Gaussian
RWM_cw <- function(logistic_posterior, nits, h, x_curr, X, Y) {
  d <- length(x_curr)
  accepted <- rep(0, d)  # Track acceptance for each parameter
  x_store <- matrix(nrow = nits, ncol = d)
  
  # Initialize current log-posterior
  logpi_curr <- logistic_posterior(x_curr, X, Y)
  if (is.nan(logpi_curr)) logpi_curr <- -Inf  # Handle NaN

  for (i in 1:nits) {
    for (j in 1:d) {
      x_prop <- x_curr
      # Propose a candidate move for the j-th parameter
      x_prop[j] <- x_curr[j] + h[j] * rnorm(1)
      
      logpi_prop <- logistic_posterior(x_prop, X, Y)
      # Check for numerical issues
      if (is.nan(logpi_prop)) logpi_prop <- -Inf  # Handle NaN
      
      # Accept-reject step
      loga <- logpi_prop - logpi_curr
      u <- runif(1)
      # Compare u < exp(loga) to avoid numerical issues
      if (!is.nan(loga) && u < exp(loga)) {
        x_curr[j] <- x_prop[j]
        logpi_curr <- logpi_prop
        accepted[j] <- accepted[j] + 1
      }
    }
    x_store[i,] <- x_curr
  }
  
  return(list(x_store = x_store, a_rate = accepted / nits))
}

h_logGaussian <- c(1.65, 1.8, 1.75, 2.4, 2.6, 0.1, 1.6, 1.8, 0.55, 1.45)  # Step sizes for each parameter

mc_logGaussian_cw<- RWM_cw(logistic_posterior_Gaussian, nits, h_logGaussian, x_curr, X, Y_L)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logGaussian_cw$a_rate)

                                # Logistic-UIP
h_logUIP <- c(1.65, 1.8, 1.75, 2.5, 2.6, 0.1, 1.6, 1.8, 0.55, 1.45)  # Step sizes for each parameter

mc_logUIP_cw <- RWM_cw(logistic_posterior_UIP, nits, h_logUIP, x_curr, X, Y_L)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_logGaussian_cw$a_rate)

                # Trace plot for Logistic-Gaussian/UIP Component-wise
mc_trace_plot(mc_logGaussian_cw, colors, "Logistic-Gaussian Component-wise RWM")
mc_trace_plot(mc_logUIP_cw, colors, "Logistic-UIP Component-wise RWM")






                              #Cauchit Case

# MODEL: Random Walk Metropolis (RWM)
                            # Cauchit-Gaussian
# Run the RWM algorithm for logistic regression with Gaussian
mc_cauGaussian_rwm <- RWM(cauchit_posterior_Gaussian, nits, h = 0.13, x_curr, X, Y_C)

# Output the acceptance rate
cat("The acceptance rate is: ", mc_cauGaussian_rwm$a_rate)

                            # Logistic-UIP
# Run the RWM algorithm for logistic regression with Gaussian
mc_cauUIP_rwm <- RWM(cauchit_posterior_UIP, nits, h = 0.07, x_curr, X, Y_C)

# Output the acceptance rate
cat("The acceptance rate is: ", mc_cauUIP_rwm$a_rate)


                   # Call the function for the Logistic-Gaussian/UIP RWM model
mc_trace_plot(mc_cauGaussian_rwm, colors, "Cauchit-Gaussian RWM")
mc_trace_plot(mc_cauUIP_rwm, colors, "Cauchit-Gaussian RWM")


# MODEL: Pre-conditioned RWM using multivariate normal proposals
                               #Cauchit-Gaussian 
# RWM algorithm function with pre-conditioning & multivariate normal proposal
RWM_pre_mnorm

# Define parameters for the initial run
nits_initial <- 20000  
burn_in <- 5000        
h_initial <- 0.003  
x_curr_initial <- rep(0, ncol(X))

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run <- RWM_pre_mnorm(cauchit_posterior_Gaussian, X, Y_C, nits_initial, diag(rep(h_initial, ncol(X))), x_curr_initial)
# Tune the step-size to achieve a 30-40% acceptance rate
initial_run$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
nits_final <- 20000
h_final <- 0.5

# Run the RWM algorithm with the estimated covariance matrix
mc_cauGaussian_pre_norm <- RWM_pre_mnorm(cauchit_posterior_Gaussian, X, Y_C, nits_final, cov_matrix_est * h_final, x_curr_initial)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ",mc_cauGaussian_pre_norm$a_rate)

                                    #Cauchit-UIP
# RWM algorithm function with pre-conditioning
RWM_pre_mnorm

# Define parameters for the initial run
nits_initial <- 20000  
burn_in <- 5000        
h_initial <- 0.0015  
x_curr_initial <- rep(0, ncol(X))

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run <- RWM_pre_mnorm(cauchit_posterior_UIP, X, Y_C, nits_initial, diag(rep(h_initial, ncol(X))), x_curr_initial)
# Tune the step-size to achieve a 30-40% acceptance rate
initial_run$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
nits_final <- 20000
h_final <- 0.45  

# Run the RWM algorithm with the estimated covariance matrix
mc_cauUIP_pre_norm <- RWM_pre_mnorm(cauchit_posterior_UIP, X, Y_C, nits_final, cov_matrix_est * h_final, x_curr_initial)


# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_cauUIP_pre_norm$a_rate)

# Trace plot for Logistic-Gaussian preconditioned_norm
mc_trace_plot(mc_cauGaussian_pre_norm, colors, "Pre Cauchit-Gaussian mnorm")
mc_trace_plot(mc_cauUIP_pre_norm, colors, "Pre Cauchit-UIP mnorm")


# MODEL: Pre-conditioned RWM using multivariate t proposals
                                #Logistic-Gaussian
# RWM algorithm function with pre-conditioning & multivariate t proposal
RWM_pre_mt
# Define parameters for the initial run
h_initial <- 0.001      # Initial proposal width, used to scale the identity covariance matrix
df_t <- 1

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run_t <- RWM_pre_mt(cauchit_posterior_Gaussian, X, Y_C, nits_initial, diag(rep(h_initial, ncol(X))), x_curr_initial, df_t)

# Tune the step-size to achieve a 30-40% acceptance rate
initial_run_t$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run_t$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
nits_final <- 20000
h_final <- 0.35

# Run the RWM algorithm with the estimated covariance matrix
mc_cauGaussian_pre_t <- RWM_pre_mt(cauchit_posterior_Gaussian, X, Y_C, nits_final, cov_matrix_est*h_final, x_curr_initial, df_t)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_cauGaussian_pre_t$a_rate)

                               #Cauchit-UIP (rmvt)
# RWM algorithm function with pre-conditioning & multivariate t proposal
RWM_pre_mt
# Define parameters for the initial run
h_initial <- 0.001      # Initial proposal width, used to scale the identity covariance matrix
df_t <- 1

# Run the initial RWM algorithm with a simple diagonal covariance matrix
initial_run_t <- RWM_pre_mt(cauchit_posterior_UIP, X, Y_C, nits_initial, diag(rep(h_initial, ncol(X))), x_curr_initial, df_t)

# Tune the step-size to achieve a 30-40% acceptance rate
initial_run_t$a_rate

# Estimate the covariance matrix from the initial run (post burn-in)
cov_matrix_est <- cov(initial_run_t$x_store[(burn_in + 1):nits_initial, ])

# Define parameters for the pre-conditioned run
nits_final <- 20000
h_final <- 0.35

# Run the RWM algorithm with the estimated covariance matrix
mc_cauUIP_pre_t <- RWM_pre_mt(cauchit_posterior_UIP, X, Y_C, nits_final, cov_matrix_est*h_final, x_curr_initial, df_t)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_cauUIP_pre_t$a_rate)

                   # Trace plot for Logistic-UIP preconditioned_norm
mc_trace_plot(mc_cauGaussian_pre_t , colors, "Pre Cauchit-Gaussian mt")
mc_trace_plot(mc_cauUIP_pre_t, colors, "Pre Cauchit-UIP mt")



# MODEL: Metropolis-adjusted Langevin Algorithm (MALA)
                              # Cauchit-Gaussian
# Calculating the gradient
dlogpi_cauGaussian <- function(beta, X, Y) {
  eta <- X %*% beta
  p <- 0.5 + (1 / pi) * atan(eta)
  dp_deta <- 1 / (pi * (1 + eta^2))  # derivative of p w.r.t eta
  grad_log_likelihood <- t(X) %*% ((Y - p) * dp_deta)  # chain rule to get the gradient w.r.t beta
  
  grad_log_prior <- -beta 
  
  return(grad_log_likelihood + grad_log_prior)
}

mc_cauGaussian_mala <- MALA(cauchit_posterior_Gaussian, dlogpi_cauGaussian, nits, h = 0.002, x_curr, X, Y_C)

                             # Performing burn-in
burn_in <- 5000  # Define the number of burn-in iterations

# Create new MCMC objects that exclude the burn-in period
mc_cauGaussian_mala_burnin <- mc_cauGaussian_mala
mc_cauGaussian_mala_burnin$x_store <- mc_cauGaussian_mala$x_store[(burn_in + 1):nrow(mc_cauGaussian_mala$x_store), ]

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_cauGaussian_mala_burnin$a_rate)

                                # Cauchit-UIP
# Calculating the gradient
dlogpi_cauUIP <- function(beta, X, Y) {
  n <- nrow(X)
  Sigma_inv <- solve(t(X) %*% X) * n
  
  eta <- X %*% beta
  p <- 0.5 + (1 / pi) * atan(eta)
  dp_deta <- 1 / (pi * (1 + eta^2))  # derivative of p w.r.t eta
  grad_log_likelihood <- t(X) %*% ((Y - p) * dp_deta)  # chain rule to get the gradient w.r.t beta
  
  grad_log_prior <- -Sigma_inv %*% beta  # derivative of UIP log-prior w.r.t beta
  
  return(grad_log_likelihood + grad_log_prior)
}


mc_cauUIP_mala <- MALA(cauchit_posterior_UIP, dlogpi_cauUIP, nits, h = 0.005, x_curr, X, Y_C)
                        # Performing burn-in
burn_in <- 5000  

# Create new MCMC objects that exclude the burn-in period
mc_cauUIP_mala_burnin <- mc_cauUIP_mala
mc_cauUIP_mala_burnin$x_store <- mc_cauUIP_mala$x_store[(burn_in + 1):nrow(mc_cauUIP_mala$x_store), ]

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_cauUIP_mala_burnin$a_rate)

                # Trace plot for Logistic-Gaussian/UIP MALA
mc_trace_plot(mc_cauGaussian_mala_burnin, colors, "Post Burn-In Logistic-Gaussian MALA")
mc_trace_plot(mc_cauUIP_mala_burnin, colors, "Post Burn-IN Logistic-UIP MALA")


# MODEL: Component-wise MCMC
                             # Cauchit-Gaussian
h <- c(2, 2.2, 2.2, 3.2, 3.4, 0.2, 2.3, 2.5, 0.9, 2.1)  # Step sizes for each parameter

mc_cauGaussian_cw<- RWM_cw(cauchit_posterior_Gaussian, nits, h, x_curr, X, Y_C)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_cauGaussian_cw$a_rate)

                                # Logistic-UIP
h <- c(1.65, 1.8, 1.75, 2.5, 2.6, 0.09, 1.6, 1.8, 0.55, 1.45)  # Step sizes for each parameter

mc_cauUIP_cw <- RWM_cw(cauchit_posterior_UIP, nits, h, x_curr, X, Y_C)

# Output the acceptance rate of the final run
cat("The acceptance rate of the final run is: ", mc_cauGaussian_cw$a_rate)

                # Trace plot for Cauchit-Gaussian/UIP Component-wise
mc_trace_plot(mc_cauGaussian_cw, colors, "Cauchit-Gaussian Component-wise RWM")
mc_trace_plot(mc_cauUIP_cw, colors, "Cauchit-UIP Component-wise RWM")





# Section 4: Model Diagnostics for MCMC Algorithms

# This section is dedicated to performing diagnostic analyses on the fitted 
# MCMC models for both Logistic and Cauchit regression cases. We employ various 
# statistical methods to assess the predictive accuracy and convergence of the 
# models. The diagnostics are as follows:

# 1. Prediction Accuracy: This involves comparing the posterior mean of the 
#    parameters against the true beta coefficients, utilizing Brier scores 
#    as a measure of predictive performance.

# 2. Convergence Diagnostics: We use Gelman-Rubin statistics to evaluate the 
#    convergence across multiple chains for each MCMC algorithm.

# 3. Autocorrelation Analysis: Autocorrelation plots are generated for each 
#    parameter to visualize the degree of correlation between MCMC samples at 
#    different lags.

# 4. Effective Sample Size (ESS): The ESS is computed for each chain to quantify 
#    the number of effectively independent samples.

# Each diagnostic is applied to the output of each MCMC algorithm for both model 
# types. Results from these diagnostics provide insights into the reliability and 
# efficiency of the algorithms in parameter estimation. The diagnostics are split 
# into separate sections for the Logistic and Cauchit cases to maintain clarity 
# and organization.


                            #Logistic Case
# Define all models directly in a list for the ease of further analysis
models_list_L <- list(
  model1_L = mc_logGaussian_rwm,
  model2_L = mc_logUIP_rwm,
  model3a_L = mc_logGaussian_pre_norm,
  model3b_L = mc_logGaussian_pre_t,
  model4a_L = mc_logUIP_pre_norm,
  model4b_L = mc_logUIP_pre_t,
  model5_L = mc_logGaussian_mala_burnin,
  model6_L = mc_logUIP_mala_burnin,
  model7_L = mc_logGaussian_cw,
  model8_L = mc_logUIP_cw
)

# Convert all model outputs to mcmc objects in a single step using lapply
mcmc_models_L <- lapply(models_list_L, function(model) mcmc(model$x_store))

# Optional: if you want to rename these mcmc objects similar to original naming
names(mcmc_models_L) <- c("mcmc_1", "mcmc_2", "mcmc_3a", "mcmc_3b", "mcmc_4a", "mcmc_4b", "mcmc_5", "mcmc_6", "mcmc_7", "mcmc_8")

# Diagnostic 1: Prediction Accuracy
# Diagnostic 1 1.1: Comparing Posterior Mean with True Beta Parameters
mcmc_objects_L <- lapply(mcmc_models_L, mcmc)
summary_stats_L <- lapply(mcmc_objects_L, summary)
print(summary_stats_L)

# Compute the matrix of errors
error_matrix <- sapply(models_list_L, function(model) {
  model_means <- colMeans(model$x_store)
  model_means - beta  
})

# Convert the error matrix to a data frame for plotting
df_errors <- melt(error_matrix)
names(df_errors) <- c("Parameter", "Model", "Error")
              #Heatmap of Estimation Error for Logistic-Gaussian
gaussian_models <- c("model1_L", "model3a_L", "model3b_L", "model5_L", "model7_L")
uip_models <- c("model2_L", "model4a_L", "model4b_L", "model6_L", "model8_L")

# Subset df_errors for Gaussian prior
df_errors_gaussian <- df_errors[df_errors$Model %in% gaussian_models,]

# Plotting Gaussian prior
ggplot(df_errors_gaussian, aes(x = Model, y = Parameter, fill = Error)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "green", mid = "white", midpoint = 0, 
                       name = "Error (True - Estimate)") +
  theme_minimal() +
  labs(title = "Heatmap of Estimation Errors for Logistic - Gaussian",
       x = "Model",
       y = "Beta Parameter") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


              #Heat map of Estimation Error for Logistic-UIP
# Subset df_errors for UIP prior
df_errors_uip <- df_errors[df_errors$Model %in% uip_models,]

# Plotting UIP prior
ggplot(df_errors_uip, aes(x = Model, y = Parameter, fill = Error)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "green", mid = "white", midpoint = 0, 
                       name = "Error (True - Estimate)") +
  theme_minimal() +
  labs(title = "Heatmap of Estimation Errors for Logistic - UIP",
       x = "Model",
       y = "Beta Parameter") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Diagnostics 1.2: Brier Score
# Define the Brier score function
brier_score <- function(Y, p) {
  mean((Y - p)^2)
}

# Write a brier calculation function to speed up the process
calculate_brier <- function(model, X, Y) {
  # Get the mean of the MCMC samples for the final beta estimates
  beta_estimates <- apply(model$x_store, 2, mean)

  # Predict probabilities
  p_pred <- 1 / (1 + exp(-X %*% beta_estimates))

  # Calculate Brier score for the logistic regression with Gaussian prior
  brier_score <- brier_score(p_pred, Y)

  # Return the Brier score
  return(brier_score)
}

brier_scores_L <- lapply(models_list_L, calculate_brier, X, Y_L)

print(unlist(brier_scores_L))


# Diagnostic 2: The Gelman-Rubin diagnostic
# Write a function for G-R statistics calculation to automate the process
gelman_rubin <- function(posterior_func, X, Y, nits, h, cov_matrix=NULL, df_t=NULL, model_type="RWM") {
  starting_values <- c(0.5, 0, -0.5)
  chains <- list()
  for (initial_value in starting_values) {
    if (model_type == "RWM") {
      mcmc_results <- RWM(posterior_func, nits, h, x_curr = rep(initial_value, ncol(X)), X, Y)
    } else if (model_type == "RWM_pre") {
      mcmc_results <- RWM_pre_mnorm(posterior_func, X, Y, nits, cov_matrix, x_curr = rep(initial_value, ncol(X)))
    } else if (model_type == "RWM_pre_mt") {
      mcmc_results <- RWM_pre_mt(posterior_func, X, Y, nits, cov_matrix, x_curr = rep(initial_value, ncol(X)), df_t)
    } else if (model_type == "RWM_cw") {
      mcmc_results <- RWM_cw(posterior_func, nits, h, x_curr = rep(initial_value, ncol(X)), X, Y)
    }
    chains[[paste("start", initial_value)]] <- mcmc(mcmc_results$x_store)
  }
  mcmc_chains <- mcmc.list(chains[[1]], chains[[2]], chains[[3]])
  return(gelman.diag(mcmc_chains))
}

# Model 1: Logistic - Gaussian RWM
gelman_rubin(logistic_posterior_Gaussian, X, Y_L, nits, 0.08)

# Model 2: Logistic - UIP RWM
gelman_rubin (logistic_posterior_UIP, X, Y_L, nits, 0.05)

# Model 3a: Logistic - Gaussian Preconditioned mnorm
gelman_rubin(logistic_posterior_Gaussian, X, Y_L, nits_final, 0.45, cov_matrix_est, model_type="RWM_pre")

# Model 3b: Logistic - Gaussian Preconditioned mt
gelman_rubin(logistic_posterior_Gaussian, X, Y_L, nits_final, 0.35, cov_matrix_est, df_t, "RWM_pre_mt")

# Model 4a: Logistic - UIP Preconditioned mnorm
gelman_rubin(logistic_posterior_UIP, X, Y_L, nits_final, 0.45, cov_matrix_est, model_type="RWM_pre")

# Model 4b: Logistic - UIP Preconditioned mt
gelman_rubin(logistic_posterior_UIP, X, Y_L, nits_final, 0.35, cov_matrix_est, df_t, "RWM_pre_mt")

# Model 7: Logistic - Gaussian Component-wise
gelman_rubin(logistic_posterior_Gaussian, X, Y_L, nits, h_logGaussian, model_type="RWM_cw")

# Model 8: Logistic - UIP Component-wise
gelman_rubin(logistic_posterior_UIP, X, Y_L, nits, h_logUIP, model_type = "RWM_cw")

# We isolate MALA because we applied burn-in, so the calculation function doesn't apply
#Model 5
# Run MCMC for each starting value and handle burn-in
chains <- list()
starting_values <- c(0.5, 0, -0.5)
for (initial_value in starting_values) {
  # Run MALA MCMC
  mcmc_results <- MALA(logistic_posterior_Gaussian, dlogpi_logGaussian, nits, h = 0.001, x_curr = rep(initial_value, ncol(X)), X, Y_L)
  
  # Apply burn-in and convert to mcmc object
  effective_samples <- mcmc_results$x_store[(burn_in + 1):nrow(mcmc_results$x_store), ]
  chains[[paste("start", initial_value)]] <- mcmc(effective_samples)
}

# Combine all the chains into an mcmc.list
mala_chains <- mcmc.list(chains[[1]], chains[[2]], chains[[3]])
gelman.diag(mala_chains)

#Model 6
# Run MCMC for each starting value and handle burn-in
chains <- list()
starting_values <- c(0.5, 0, -0.5)
for (initial_value in starting_values) {
  # Run MALA MCMC
  mcmc_results <- MALA(logistic_posterior_UIP, dlogpi_logUIP, nits, h = 0.001, x_curr = rep(initial_value, ncol(X)), X, Y_L)
  
  # Apply burn-in and convert to mcmc object
  effective_samples <- mcmc_results$x_store[(burn_in + 1):nrow(mcmc_results$x_store), ]
  chains[[paste("start", initial_value)]] <- mcmc(effective_samples)
}

# Combine all the chains into an mcmc.list
mala_chains <- mcmc.list(chains[[1]], chains[[2]], chains[[3]])
gelman.diag(mala_chains)


# Diagnostic 3: Autocorrelation Plots
                             #Auto-correlation Plots
                    #Compare different MCMC algorithms for Gaussian prior
# Create a combined autocorrelation plot
plot(autocorr(mcmc_models_L$mcmc_1[,1], lags = 1:50), type="b", lwd=2, col="red", pch=16, ylim=c(-0.2, 1), xlab="Index", ylab="Autocorrelation", main="Autocorrelation for Beta 1 across Models for Logistic-Gaussian")
lines(autocorr(mcmc_models_L$mcmc_3a[,1], lags = 1:50), type="b", lwd=2, col="blue", pch=16)
lines(autocorr(mcmc_models_L$mcmc_3b[,1], lags = 1:50), type="b", lwd=2, col="pink", pch=16)
lines(autocorr(mcmc_models_L$mcmc_5[,1], lags = 1:50), type="b", lwd=2, col="green", pch=16)
lines(autocorr(mcmc_models_L$mcmc_7[,1], lags = 1:50), type="b", lwd=2, col="yellow", pch=16)

abline(h = -0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0, lty = 3, lwd = 3, col = "red")

legend(x = 35, y = 0.7,  
       legend=c("vanilla RWM", "preconditioned RWM mnorm", "preconditioned RWM mt", "MALA","component RWM"), col=c("red", "blue", "pink", "green","yellow"), lwd=2,cex = 0.6, pch=16,  xpd = TRUE)

                        #Compare different MCMC algorithms for UIP prior
# Create a combined autocorrelation plot
plot(autocorr(mcmc_models_L$mcmc_2[,1], lags = 1:50), type="b", lwd=2, col="red", pch=16, ylim=c(-0.2, 1), xlab="Index", ylab="Autocorrelation", main="Autocorrelation for Beta 1 across Models for Logistic-UIP")
lines(autocorr(mcmc_models_L$mcmc_4a[,1], lags = 1:50), type="b", lwd=2, col="blue", pch=16)
lines(autocorr(mcmc_models_L$mcmc_4b[,1], lags = 1:50), type="b", lwd=2, col="pink", pch=16)
lines(autocorr(mcmc_models_L$mcmc_6[,1], lags = 1:50), type="b", lwd=2, col="green", pch=16)
lines(autocorr(mcmc_models_L$mcmc_8[,1], lags = 1:50), type="b", lwd=2, col="yellow", pch=16)

abline(h = -0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0, lty = 3, lwd = 3, col = "red")

legend(x = 35, y = 0.7,  
       legend=c("vanilla RWM", "preconditioned RWM mnorm", "preconditioned RWM mt", "MALA","component RWM"), 
       col=c("red", "blue", "pink", "green","yellow"), lwd=2,cex = 0.6, pch=16,  xpd = TRUE)
                        #Heatmap for all beta paramters
                        #Model 3a 
# Assuming mcmc_object is your MCMC output as an mcmc object
compute_autocorrelations <- function(mcmc_object, max_lag) {
    autocorr_data <- sapply(1:ncol(mcmc_object), function(i) {
        autocorr(mcmc_object[, i], lags = 1:max_lag)
    })
    autocorr_data
}

# Function to plot heatmap of autocorrelations
plot_autocorrelation_heatmap <- function(mcmc_object, max_lag = 50, title = "Autocorrelation Heatmap") {
    # Compute autocorrelations
    autocorr_matrix <- compute_autocorrelations(mcmc_object, max_lag)
    
    # Convert matrix to data frame for plotting
    autocorr_df <- melt(autocorr_matrix)
    colnames(autocorr_df) <- c("Lag", "Parameter", "Autocorrelation")
    
    # Convert parameter index to factor for better plotting
    autocorr_df$Parameter <- as.factor(autocorr_df$Parameter)
    
    # Create the heatmap plot
    heatmap_plot <- ggplot(autocorr_df, aes(x = Lag, y = Parameter, fill = Autocorrelation)) +
        geom_tile() + 
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), 
                             space = "Lab", name="Autocorrelation") +
        theme_minimal() +
        labs(title = title, x = "Lag", y = "Parameter")
    
    # Display the plot
    print(heatmap_plot)
}

plot_autocorrelation_heatmap(mcmc_models_L$mcmc_3a, max_lag = 50, 
                             title = "Autocorrelation Heatmap of Logistic-Gaussian Preconditioned mnorm")
plot_autocorrelation_heatmap(mcmc_models_L$mcmc_7, max_lag = 50,
                             title = "Autocorrelation Heatmap of Logistic-Gaussian Component-wise RWM")

plot_autocorrelation_heatmap(mcmc_models_L$mcmc_4a, max_lag = 50, 
                             title = "Autocorrelation Heatmap of Logistic-UIP Preconditioned mnorm")
plot_autocorrelation_heatmap(mcmc_models_L$mcmc_8, max_lag = 50, 
                             title = "Autocorrelation Heatmap of Logistic-UIP Component-wise RWM")



# Diagnostic 4: Effective Sample Size
# Compute ESS for each model using lapply
ess_values_L <- lapply(mcmc_models_L, effectiveSize)

# Calculate descriptive statistics for each model's ESS
ess_stats_L <- sapply(ess_values_L, function(ess) {
  c(Mean = mean(ess), Median = median(ess), Min = min(ess), Max = max(ess))
})

print(ess_stats_L)


                                 #Cauchit Case
# Define all Cauchit models directly in a list
models_list_C <- list(
  model1_C = mc_cauGaussian_rwm,
  model2_C = mc_cauUIP_rwm,
  model3a_C = mc_cauGaussian_pre_norm,
  model3b_C = mc_cauGaussian_pre_t,
  model4a_C = mc_cauUIP_pre_norm,
  model4b_C = mc_cauUIP_pre_t,
  model5_C = mc_cauGaussian_mala_burnin,
  model6_C = mc_cauUIP_mala_burnin,
  model7_C = mc_cauGaussian_cw,
  model8_C = mc_cauUIP_cw
)

# Convert all Cauchit model outputs to mcmc objects in a single step using lapply
mcmc_models_C <- lapply(models_list_C, function(model) mcmc(model$x_store))

# Optional: Rename these mcmc objects similar to original naming for consistency
names(mcmc_models_C) <- c("mcmc_1", "mcmc_2", "mcmc_3a", "mcmc_3b", "mcmc_4a", "mcmc_4b", "mcmc_5", "mcmc_6", "mcmc_7", "mcmc_8")


# Diagnostic 1: Prediction Accuracy
# Diagnostic 1.1 Comparing Posterior Mean with True Beta Parameters

mcmc_objects_C <- lapply(mcmc_models_C, mcmc)
summary_stats_C <- lapply(mcmc_objects_C, summary)

# Compute the matrix of errors
error_matrix <- sapply(models_list_C, function(model) {
  model_means <- colMeans(model$x_store)
  model_means - beta  
})

# Convert the error matrix to a data frame for plotting
df_errors <- melt(error_matrix)
names(df_errors) <- c("Parameter", "Model", "Error")

              #Heatmap of Estimation Error for Cauchit-Gaussian
gaussian_models <- c("model1_C", "model3a_C", "model3b_C", "model5_C", "model7_C")
uip_models <- c("model2_C", "model4a_C", "model4b_C", "model6_C", "model8_C")

# Subset df_errors for Gaussian prior
df_errors_gaussian <- df_errors[df_errors$Model %in% gaussian_models,]

# Plotting Gaussian prior
ggplot(df_errors_gaussian, aes(x = Model, y = Parameter, fill = Error)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "green", mid = "white", midpoint = 0, 
                       name = "Error (True - Estimate)") +
  theme_minimal() +
  labs(title = "Heatmap of Estimation Errors for Cauchit - Gaussian",
       x = "Model",
       y = "Beta Parameter") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


              #Heatmap of Estimation Error for Cauchit-UIP
# Subset df_errors for UIP prior
df_errors_uip <- df_errors[df_errors$Model %in% uip_models,]

# Plotting UIP prior
ggplot(df_errors_uip, aes(x = Model, y = Parameter, fill = Error)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "green", mid = "white", midpoint = 0, 
                       name = "Error (True - Estimate)") +
  theme_minimal() +
  labs(title = "Heatmap of Estimation Errors for Cauchit - UIP",
       x = "Model",
       y = "Beta Parameter") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Diagnostic 1.2 Brier Score

# Define the Brier score function
brier_score <- function(Y, p) {
  mean((Y - p)^2)
}

calculate_brier <- function(model, X, Y) {
  # Get the mean of the MCMC samples for the final beta estimates
  beta_estimates <- apply(model$x_store, 2, mean)

  # Predict probabilities
  p_pred <- 1 / (1 + exp(-X %*% beta_estimates))

  # Calculate Brier score for the logistic regression with Gaussian prior
  brier_score <- brier_score(p_pred, Y)

  # Return the Brier score
  return(brier_score)
}

brier_scores_C <- lapply(models_list_C, calculate_brier, X, Y_C)

print(unlist(brier_scores_C))

# Diagnostic 2: The Gelman-Rubin diagnostic

gelman_rubin <- function(posterior_func, X, Y, nits, h, cov_matrix=NULL, df_t=NULL, model_type="RWM") {
  starting_values <- c(0.5, 0, -0.5)
  chains <- list()
  for (initial_value in starting_values) {
    if (model_type == "RWM") {
      mcmc_results <- RWM(posterior_func, nits, h, x_curr = rep(initial_value, ncol(X)), X, Y)
    } else if (model_type == "RWM_pre") {
      mcmc_results <- RWM_pre_mnorm(posterior_func, X, Y, nits, cov_matrix, x_curr = rep(initial_value, ncol(X)))
    } else if (model_type == "RWM_pre_mt") {
      mcmc_results <- RWM_pre_mt(posterior_func, X, Y, nits, cov_matrix, x_curr = rep(initial_value, ncol(X)), df_t)
    } else if (model_type == "RWM_cw") {
      mcmc_results <- RWM_cw(posterior_func, nits, h, x_curr = rep(initial_value, ncol(X)), X, Y)
    }
    chains[[paste("start", initial_value)]] <- mcmc(mcmc_results$x_store)
  }
  mcmc_chains <- mcmc.list(chains[[1]], chains[[2]], chains[[3]])
  return(gelman.diag(mcmc_chains))
}

# Model 1: Cauchit - Gaussian RWM
gelman_rubin(cauchit_posterior_Gaussian, X, Y_C, nits, 0.08)

# Model 2: Cauchit - UIP RWM
gelman_rubin(cauchit_posterior_UIP, X, Y_C, nits, 0.05)

# Model 3a: Cauchit - Gaussian Preconditioned mnorm
gelman_rubin(cauchit_posterior_Gaussian, X, Y_C, nits_final, 0.45, cov_matrix_est, model_type="RWM_pre")

# Model 3b: Cauchit - Gaussian Preconditioned mt
gelman_rubin(cauchit_posterior_Gaussian, X, Y_C, nits_final, 0.35, cov_matrix_est, df_t, "RWM_pre_mt")

# Model 4a: Cauchit - UIP Preconditioned mnorm
gelman_rubin(cauchit_posterior_UIP, X, Y_C, nits_final, 0.45, cov_matrix_est, model_type="RWM_pre")

# Model 4b: Cauchit - UIP Preconditioned mt
gelman_rubin(cauchit_posterior_UIP, X, Y_C, nits_final, 0.35, cov_matrix_est, df_t, "RWM_pre_mt")

# Model 7: Cauchit - Gaussian Component-wise
gelman_rubin(cauchit_posterior_Gaussian, X, Y_C, nits, h_logGaussian, model_type="RWM_cw")

# Model 8: Cauchit - UIP Component-wise
gelman_rubin(cauchit_posterior_UIP, X, Y_C, nits, h_logUIP, model_type = "RWM_cw")

#Model 5
# Run MCMC for each starting value and handle burn-in
chains <- list()
starting_values <- c(0.5, 0, -0.5)
for (initial_value in starting_values) {
  # Run MALA MCMC
  mcmc_results <- MALA(cauchit_posterior_Gaussian, dlogpi_cauGaussian, nits, h = 0.001, x_curr = rep(initial_value, ncol(X)), X, Y_C)
  
  # Apply burn-in and convert to mcmc object
  effective_samples <- mcmc_results$x_store[(burn_in + 1):nrow(mcmc_results$x_store), ]
  chains[[paste("start", initial_value)]] <- mcmc(effective_samples)
}

# Combine all the chains into an mcmc.list
mala_chains <- mcmc.list(chains[[1]], chains[[2]], chains[[3]])
gelman.diag(mala_chains)

#Model 6
# Run MCMC for each starting value and handle burn-in
chains <- list()
starting_values <- c(0.5, 0, -0.5)
for (initial_value in starting_values) {
  # Run MALA MCMC
  mcmc_results <- MALA(cauchit_posterior_UIP, dlogpi_cauUIP, nits, h = 0.001, x_curr = rep(initial_value, ncol(X)), X, Y_C)
  
  # Apply burn-in and convert to mcmc object
  effective_samples <- mcmc_results$x_store[(burn_in + 1):nrow(mcmc_results$x_store), ]
  chains[[paste("start", initial_value)]] <- mcmc(effective_samples)
}

# Combine all the chains into an mcmc.list
mala_chains <- mcmc.list(chains[[1]], chains[[2]], chains[[3]])
gelman.diag(mala_chains)

# Diagnostic 3: Autocorrelation Plots
                          #Auto-correlation Plots
                    #Compare different MCMC algorithms for Gaussian prior
# Create a combined autocorrelation plot
plot(autocorr(mcmc_models_C$mcmc_1[,1], lags = 1:50), type="b", lwd=2, col="red", pch=16, ylim=c(-0.2, 1), xlab="Index", ylab="Autocorrelation", main="Autocorrelation for Beta 1 across Models for Cauchit-Gaussian")
lines(autocorr(mcmc_models_C$mcmc_3a[,1], lags = 1:50), type="b", lwd=2, col="blue", pch=16)
lines(autocorr(mcmc_models_C$mcmc_3b[,1], lags = 1:50), type="b", lwd=2, col="pink", pch=16)
lines(autocorr(mcmc_models_C$mcmc_5[,1], lags = 1:50), type="b", lwd=2, col="green", pch=16)
lines(autocorr(mcmc_models_C$mcmc_7[,1], lags = 1:50), type="b", lwd=2, col="yellow", pch=16)

abline(h = -0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0, lty = 3, lwd = 3, col = "red")

legend(x = 35, y = 0.7,  
       legend=c("vanilla RWM", "preconditioned RWM mnorm", "preconditioned RWM mt", "MALA","component RWM"), col=c("red", "blue", "pink", "green","yellow"), lwd=2,cex = 0.5, pch=16,  xpd = TRUE)

                        #Compare different MCMC algorithms for UIP prior
# Create a combined autocorrelation plot
plot(autocorr(mcmc_models_C$mcmc_2[,1], lags = 1:50), type="b", lwd=2, col="red", pch=16, ylim=c(-0.2, 1), xlab="Index", ylab="Autocorrelation", main="Autocorrelation for Beta 1 across Models for Cauchit-UIP")
lines(autocorr(mcmc_models_C$mcmc_4a[,1], lags = 1:50), type="b", lwd=2, col="blue", pch=16)
lines(autocorr(mcmc_models_C$mcmc_4b[,1], lags = 1:50), type="b", lwd=2, col="pink", pch=16)
lines(autocorr(mcmc_models_C$mcmc_6[,1], lags = 1:50), type="b", lwd=2, col="green", pch=16)
lines(autocorr(mcmc_models_C$mcmc_8[,1], lags = 1:50), type="b", lwd=2, col="yellow", pch=16)

abline(h = -0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0.1, lty = 3, lwd = 2, col = "red")
abline(h = 0, lty = 3, lwd = 3, col = "red")

legend(x = 35, y = 0.7,  
       legend=c("vanilla RWM", "preconditioned RWM mnorm", "preconditioned RWM mt", "MALA","component RWM"), col=c("red", "blue", "pink", "green","yellow"), lwd=2,cex = 0.5, pch=16,  xpd = TRUE)


                        #Heatmap for all beta paramters
                        #Model 3a 
# Assuming mcmc_object is your MCMC output as an mcmc object
compute_autocorrelations <- function(mcmc_object, max_lag) {
    autocorr_data <- sapply(1:ncol(mcmc_object), function(i) {
        autocorr(mcmc_object[, i], lags = 1:max_lag)
    })
    autocorr_data
}

# Function to plot heatmap of autocorrelations
plot_autocorrelation_heatmap <- function(mcmc_object, max_lag = 50, title = "Autocorrelation Heatmap") {
    # Compute autocorrelations
    autocorr_matrix <- compute_autocorrelations(mcmc_object, max_lag)
    
    # Convert matrix to data frame for plotting
    autocorr_df <- melt(autocorr_matrix)
    colnames(autocorr_df) <- c("Lag", "Parameter", "Autocorrelation")
    
    # Convert parameter index to factor for better plotting
    autocorr_df$Parameter <- as.factor(autocorr_df$Parameter)
    
    # Create the heatmap plot
    heatmap_plot <- ggplot(autocorr_df, aes(x = Lag, y = Parameter, fill = Autocorrelation)) +
        geom_tile() + 
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name="Autocorrelation") +
        theme_minimal() +
        labs(title = title, x = "Lag", y = "Parameter")
    
    # Display the plot
    print(heatmap_plot)
}

plot_autocorrelation_heatmap(mcmc_models_C$mcmc_3a, max_lag = 50, title = "Autocorrelation Heatmap of Cauchit-Gaussian Preconditioned mnorm")
plot_autocorrelation_heatmap(mcmc_models_C$mcmc_7, max_lag = 50, title = "Autocorrelation Heatmap of Cauchit-Gaussian Component-wise RWM")

plot_autocorrelation_heatmap(mcmc_models_C$mcmc_4a, max_lag = 50, title = "Autocorrelation Heatmap of Cauchit-UIP Preconditioned mnorm")
plot_autocorrelation_heatmap(mcmc_models_C$mcmc_8, max_lag = 50, title = "Autocorrelation Heatmap of Cauchit-UIP Component-wise RWM")


# Diagnostic 4: Effective Sample Size

# Compute ESS for each model using lapply
ess_values_C <- lapply(mcmc_models_C, effectiveSize)

# Calculate descriptive statistics for each model's ESS
ess_stats_C <- sapply(ess_values_C, function(ess) {
  c(Mean = mean(ess), Median = median(ess), Min = min(ess), Max = max(ess))
})

print(ess_stats_C)
