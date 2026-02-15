# ==============================================================================
# PROJECT: Bayesian Inference for Logistic Regression (Diabetes)
# FILE: main_analysis.R
# Author : Harsh Rana (229989) & Raj Pawar (231811)
# ==============================================================================

# SETUP AND LIBRARIES
if(!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, MASS, coda, caret, mvtnorm, knitr)

data_path <- here::here("Data", "diabetes.csv")
img_path  <- here::here("Report", "images")

# Create image directory if it doesn't exist
if(!dir.exists(img_path)) dir.create(img_path, recursive = TRUE)

# DATA LOADING & PREPROCESSING 
cat("Loading Data...\n")
data <- read_csv(data_path, show_col_types = FALSE)

# Standardize predictors (Glucose, BMI, etc.) for better MCMC convergence
# We scale columns 1 to 8 (Predictors). Column 9 is Outcome.
data_scaled <- data
data_scaled[,1:8] <- scale(data[,1:8])

# Train/Test Split (80% Train, 20% Test)
set.seed(2026)
trainIndex <- createDataPartition(data_scaled$Outcome, p = .8, list = FALSE, times = 1)
dataTrain <- data_scaled[ trainIndex,]
dataTest  <- data_scaled[-trainIndex,]

# Create Matrices for Matrix Multiplication
X_train <- as.matrix(cbind(1, dataTrain[, -9])) # Add intercept (column of 1s)
y_train <- dataTrain$Outcome
X_test  <- as.matrix(cbind(1, dataTest[, -9]))
y_test  <- dataTest$Outcome

# FREQUENTIST MODEL (The "Control" Group)
# We run this to compare our Bayesian results against standard MLE.
cat("Running Frequentist GLM...\n")
freq_model <- glm(Outcome ~ ., data = dataTrain, family = binomial)
freq_est   <- coef(freq_model)
freq_ci    <- confint(freq_model) # Standard 95% Confidence Intervals

# BAYESIAN MODEL (The "Experimental" Group) 
# Log-Posterior Function
log_posterior <- function(beta, X, y) {
  # Prior: Beta ~ N(0, 100*I) -> Weakly informative
  sigma_prior_sq <- 100 
  Sigma0_inv <- diag(ncol(X)) / sigma_prior_sq
  
  # Likelihood
  eta <- X %*% beta
  pi_val <- plogis(eta) # Inverse logit
  
  # Safe log-likelihood to avoid log(0)
  epsilon <- 1e-10
  log_lik <- sum(y * log(pi_val + epsilon) + (1 - y) * log(1 - pi_val + epsilon))
  
  # Log-Prior
  log_prior <- -0.5 * t(beta) %*% Sigma0_inv %*% beta
  
  return(as.numeric(log_lik + log_prior))
}

# Metropolis-Hastings Sampler
run_mh_sampler <- function(N_iter, X, y, prop_cov, init_state = NULL) {
  p <- ncol(X)
  samples <- matrix(NA, nrow = N_iter, ncol = p)
  colnames(samples) <- colnames(X)
  
  # Initial State
  if (is.null(init_state)) {
    beta_curr <- rep(0, p)
  } else {
    beta_curr <- init_state
  }
  
  log_post_curr <- log_posterior(beta_curr, X, y)
  accept_count <- 0
  
  # Loop
  for (i in 1:N_iter) {
    # Propose new beta from multivariate normal
    beta_prop <- rmvnorm(1, mean = beta_curr, sigma = prop_cov)
    
    # Calculate acceptance probability
    log_post_prop <- log_posterior(t(beta_prop), X, y)
    log_alpha <- log_post_prop - log_post_curr
    
    # Accept or Reject
    if (log(runif(1)) < log_alpha) {
      beta_curr <- beta_prop
      log_post_curr <- log_post_prop
      accept_count <- accept_count + 1
    }
    samples[i, ] <- beta_curr
  }
  
  return(list(samples = samples, accept_rate = accept_count / N_iter))
}

# Run Simulation (Multiple Chains for Diagnostics)
N_iter <- 50000
burn_in <- 10000
step_size <- 0.003 # Tuned for ~30-50% acceptance
prop_cov <- diag(ncol(X_train)) * step_size
n_chains <- 3

cat(sprintf("Running %d MCMC Chains...\n", n_chains))
set.seed(2025)

# Initialize a list to hold the coda mcmc objects
chain_list <- list()

for (c in 1:n_chains) {
  cat("  Running Chain", c, "...\n")
  # Start each chain at a randomly jittered dispersed location
  initial_beta <- rnorm(ncol(X_train), mean = 0, sd = 2) 
  
  results <- run_mh_sampler(N_iter, X_train, y_train, prop_cov, init_state = initial_beta)
  
  # Remove burn-in and convert to an mcmc object for coda
  mcmc_samples <- results$samples[(burn_in + 1):N_iter, ]
  chain_list[[c]] <- mcmc(mcmc_samples)
}

# Create an mcmc.list required by coda for multi-chain diagnostics
mcmc_chains <- mcmc.list(chain_list)

cat("\n--- CONVERGENCE DIAGNOSTICS ---\n")

# Gelman-Rubin Diagnostic (R-hat)
gelman_result <- gelman.diag(mcmc_chains)
print(gelman_result)

# Effective Sample Size (ESS)
ess_result <- effectiveSize(mcmc_chains)
cat("\nEffective Sample Size:\n")
print(ess_result)

# Combine all post-burn-in samples from all chains for final plotting and tables
combined_samples <- do.call(rbind, chain_list)
mcmc_samples <- combined_samples # Overwrite so existing plot code works

# PLOTTING & EXPORTING
cat("Generating Academic-Grade Plots with Large Fonts...\n")

clean_names <- c("Intercept", "Pregnancies", "Glucose", "Blood Pressure", 
                 "Skin Thickness", "Insulin", "BMI", "Pedigree Func.", "Age")

# Trace Plots
pdf(file.path(img_path, "trace_plots.pdf"), width = 12, height = 10)
par(mfrow=c(3,3), mar=c(6, 6, 5, 2) + 0.1) 
for(i in 1:ncol(X_train)) {
  plot(mcmc_samples[,i], type="l", main=clean_names[i], 
       ylab="Coefficient Value", xlab="Iteration", col="gray", 
       cex.main=2.0, cex.lab=1.8, cex.axis=1.5)
  
  lines(lowess(mcmc_samples[,i]), col="blue", lwd=3)
}
dev.off()

# Posterior Density Plots vs Frequentist
pdf(file.path(img_path, "posterior_density.pdf"), width = 12, height = 10)
par(mfrow=c(3,3), mar=c(6, 6, 5, 2) + 0.1)
for(i in 1:ncol(X_train)) {
  d <- density(mcmc_samples[,i])
  
  plot(d, main=clean_names[i], col="blue", lwd=3, 
       xlab="Beta Value", ylab="Density", ylim=c(0, max(d$y)*1.5),
       cex.main=2.0,   
       cex.lab=1.8,    
       cex.axis=1.5)   
  
  abline(v = freq_est[i], col="red", lwd=2, lty=2)
  
  if(i==1) {
    legend("topright", legend=c("Bayesian Posterior", "Freq. MLE"), 
           col=c("blue", "red"), lty=c(1,2), lwd=c(3,2), cex=1.3, bty="n")
  }
}
dev.off()


# Comparison Table
bayes_mean <- colMeans(mcmc_samples)
bayes_ci <- apply(mcmc_samples, 2, quantile, probs = c(0.025, 0.975))

comp_table <- data.frame(
  Parameter = colnames(X_train),
  Freq_Est = round(freq_est, 3),
  Freq_CI_Low = round(freq_ci[,1], 3),
  Freq_CI_High = round(freq_ci[,2], 3),
  Bayes_Mean = round(bayes_mean, 3),
  Bayes_CI_Low = round(bayes_ci[1,], 3),
  Bayes_CI_High = round(bayes_ci[2,], 3)
)

write_csv(comp_table, file.path(img_path, "comparison_table.csv"))

# Predictive Performance (Test Set)
# Calculate probability for every test patient using posterior means
test_logits <- X_test %*% bayes_mean
test_probs <- plogis(test_logits)
pred_classes <- ifelse(test_probs > 0.5, 1, 0)

# Confusion Matrix
cm <- confusionMatrix(as.factor(pred_classes), as.factor(y_test))

sink(file.path(img_path, "accuracy_metrics.txt"))
print(cm)
sink()

cat("Analysis Complete. All files saved to 'Report/images'.\n")