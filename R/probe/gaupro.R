# CIP-0007 phase 0: probe GauPro against the wire reference.
# Run: R_LIBS_USER=$PWD/.rlib Rscript R/probe/gaupro.R
suppressPackageStartupMessages(library(GauPro))
source("R/probe/wire.R")

# Wire lengthscale -> GauPro beta (theta = 10^beta). Verified against the wire Gram to 1e-16.
beta_gaussian <- function(l) -log10(2 * l^2)   # k = s2 exp(-theta r^2), theta = 1 / (2 l^2)
beta_matern   <- function(l) -2 * log10(l)

fixed_model <- function(X, z, kernel, trend, noise, s2)
  GauPro_kernel_model$new(X = X, Z = z, kernel = kernel, trend = trend,
                          nug = noise / s2,            # nugget is relative: K = s2 (R + nug I)
                          nug.est = FALSE, nug.min = 1e-12, param.est = FALSE, normalize = FALSE)

report <- function(name, mod, ll_shift, Xn, m, s, ref) {
  p <- mod$predict(Xn, se.fit = TRUE)
  cat(sprintf("%-16s dll=%.2e dmean=%.2e dvar(noisy)=%.2e\n", name,
              mod$loglikelihood() - ll_shift - ref$log_likelihood, max(abs(p$mean * s + m - ref$mean)),
              max(abs(p$s2 * s^2 - (ref$var_latent + ref$noise)))))   # predictive variance includes noise
}

d <- read_golden("ftc_standardize")
X <- as_matrix(d$data$X); y <- as_matrix(d$data$Y)[, 1]; Xn <- as_matrix(d$`x-reference`$Xnew)
k <- d$kernel; m <- d$normalizer$mean; s <- d$normalizer$std; z <- (y - m) / s
kk <- Matern32$new(D = 1, beta = beta_matern(k$lengthscale), s2 = k$variance, s2_est = FALSE, beta_est = FALSE)
mod <- fixed_model(X, z, kk, trend_0$new(), d$likelihood$variance, k$variance)
report("ftc_standardize", mod, length(z) * log(s), Xn, m, s, wire_reference(d))   # shift: -n log s Jacobian

d <- read_golden("ftc_rbf_white")                 # white folded into noise, mean.constant exercised
d$mean <- list(type = "mean.constant", value = 1.5)
X <- as_matrix(d$data$X); y <- as_matrix(d$data$Y)[, 1]; Xn <- as_matrix(d$`x-reference`$Xnew)
k <- d$kernel$parts[[1]]; noise <- d$likelihood$variance + d$kernel$parts[[2]]$variance
kk <- Gaussian$new(D = 1, beta = beta_gaussian(k$lengthscale), s2 = k$variance, s2_est = FALSE, beta_est = FALSE)
mod <- fixed_model(X, y, kk, trend_c$new(m = 1.5, m_est = FALSE), noise, k$variance)
ref <- wire_reference(d)
p <- mod$predict(Xn, se.fit = TRUE)
cat(sprintf("%-16s dll=%.2e dmean=%.2e dvar(noisy)=%.2e\n", "rbf_white+mean",
            mod$loglikelihood() - ref$log_likelihood, max(abs(p$mean - ref$mean)),
            max(abs(p$s2 - (ref$var_latent + ref$noise)))))
