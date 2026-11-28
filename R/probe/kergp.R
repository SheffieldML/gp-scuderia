# CIP-0007 phase 0: probe kergp against the wire reference.
# Run: R_LIBS_USER=$PWD/.rlib Rscript R/probe/kergp.R
suppressPackageStartupMessages(library(kergp))
source("R/probe/wire.R")

report <- function(name, ll, mean, var_latent, ref)
  cat(sprintf("%-18s dll=%.2e dmean=%.2e dvar(latent)=%.2e\n", name,
              ll - ref$log_likelihood, max(abs(mean - ref$mean)), max(abs(var_latent - ref$var_latent))))

# Full Gaussian log-likelihood from the pieces gp(estim = FALSE) returns (it leaves logLik NA).
loglik_gls <- function(fit) {
  n <- length(fit$y)
  -0.5 * (n * log(2 * pi) + 2 * sum(log(diag(fit$L))) + as.numeric(fit$sseStar))
}

# 1. matern32 with standardise: parameters live on the normalised scale.
d <- read_golden("ftc_standardize")
X <- as_matrix(d$data$X); y <- as_matrix(d$data$Y)[, 1]; Xn <- as_matrix(d$`x-reference`$Xnew)
k <- d$kernel; m <- d$normalizer$mean; s <- d$normalizer$std; z <- (y - m) / s
cv <- covRadial(k1Fun1 = k1Fun1Matern3_2, cov = "homo", d = 1, inputs = "x", par = c(k$lengthscale, k$variance))
fit <- gp(z ~ 1, data = data.frame(x = X[, 1], z = z), inputs = "x", cov = cv, estim = FALSE,
          varNoise = d$likelihood$variance, beta = 0)
p <- predict(fit, data.frame(x = Xn[, 1]), type = "SK", checkNames = FALSE)  # keep forceInterp = FALSE
ref <- wire_reference(d)
report("ftc_standardize", loglik_gls(fit) - length(z) * log(s), p$mean * s + m, p$sd^2 * s^2, ref)

# 2. add(prod(matern52, rbf), bias) in two dimensions.
d <- read_golden("ftc_prod_matern52")
X <- as_matrix(d$data$X); y <- as_matrix(d$data$Y)[, 1]; Xn <- as_matrix(d$`x-reference`$Xnew)
inp <- c("x1", "x2"); colnames(X) <- inp; colnames(Xn) <- inp
pt <- d$kernel$parts; m52 <- pt[[1]]$parts[[1]]; rb <- pt[[1]]$parts[[2]]; bi <- pt[[2]]
a <- covRadial(k1Fun1 = k1Fun1Matern5_2, cov = "homo", d = 2, inputs = inp, iso = 1,
               par = c(m52$lengthscale, m52$variance * rb$variance))     # product variance carried once
b <- covRadial(k1Fun1 = k1Fun1Gauss, cov = "corr", d = 2, inputs = inp, iso = 1, par = rb$lengthscale)
cc <- covMan(kernel = function(x1, x2, par) par[1], d = 2, inputs = inp, parNames = "v",
             par = bi$variance, label = "bias")
cv <- covComp(formula = ~ a() * b() + cc(), where = environment())
fit <- gp(y ~ 1, data = data.frame(X, y = y), inputs = inp, cov = cv, estim = FALSE,
          varNoise = d$likelihood$variance, beta = 0)
p <- predict(fit, Xn, type = "SK", checkNames = FALSE)
report("ftc_prod_matern52", loglik_gls(fit), p$mean, p$sd^2, wire_reference(d))
