# CIP-0007 phase 0: probe DiceKriging against the wire reference.
# Run: R_LIBS_USER=$PWD/.rlib Rscript R/probe/dicekriging.R
suppressPackageStartupMessages(library(DiceKriging))
source("R/probe/wire.R")

probe_gaussian <- function(name, covtype, fold_white = FALSE) {
  d <- read_golden(name)
  X <- as_matrix(d$data$X); y <- as_matrix(d$data$Y)[, 1]; Xn <- as_matrix(d$`x-reference`$Xnew)
  k <- d$kernel
  if (fold_white) { w <- k$parts[[2]]$variance; k <- k$parts[[1]] } else w <- 0
  m <- 0; s <- 1
  if (!is.null(d$normalizer$standardize) || !is.null(d$normalizer$std)) { m <- d$normalizer$mean; s <- d$normalizer$std }
  z <- (y - m) / s; n <- length(z)
  noise <- d$likelihood$variance + w  # wire: parameters live on the normalised scale
  mod <- km(~1, design = data.frame(x = X[, 1]), response = z, covtype = covtype, coef.trend = 0,
            noise.var = rep(noise, n), optim.method = "BFGS", parinit = k$lengthscale,
            control = list(trace = FALSE))
  # km with every parameter given does no estimation, so it is the object to predict from;
  # the fitted model above is only a carrier for logLikFun, which evaluates the full
  # Gaussian log-likelihood (known trend, noise.var) at the supplied parameters.
  ll <- logLikFun(c(k$lengthscale, k$variance), mod) - n * log(s)
  fixed <- km(~1, design = data.frame(x = X[, 1]), response = z, covtype = covtype, coef.trend = 0,
              coef.cov = k$lengthscale, coef.var = k$variance, noise.var = rep(noise, n))
  p <- predict(fixed, data.frame(x = Xn[, 1]), type = "SK", checkNames = FALSE)
  ref <- wire_reference(d)
  cat(sprintf("%-16s dll=%.2e dmean=%.2e dvar(latent+noise)=%.2e\n", name,
              ll - ref$log_likelihood, max(abs(p$mean * s + m - ref$mean)),
              max(abs(p$sd^2 * s^2 + (d$likelihood$variance + w) * s^2 - (ref$var_latent + ref$noise)))))
}
probe_gaussian("ftc_standardize", "matern3_2")
probe_gaussian("ftc_rbf_white", "gauss", fold_white = TRUE)  # white folded into noise.var
