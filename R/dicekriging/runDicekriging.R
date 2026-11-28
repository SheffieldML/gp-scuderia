# DiceKriging runner (CIP-0007). Exact GP regression with fixed parameters.
# Run by the harness with GPSCUDERIA_TASK and GPSCUDERIA_OUT set.
root <- Sys.getenv("GPSCUDERIA_ROOT")
source(file.path(root, "R", "gpscuderia", "run.R"))
source(file.path(root, "R", "gpscuderia", "gaussian.R"))
suppressPackageStartupMessages(library(DiceKriging))

COVTYPE <- c(kern.rbf = "gauss", kern.matern32 = "matern3_2", kern.matern52 = "matern5_2")
WHY_NOT <- list(
  kern.white = "kern.white: DiceKriging has no white kernel, and folding it into the noise would change the model's parameters",
  kern.bias = "kern.bias: DiceKriging has no additive constant kernel",
  kern.add = "kern.add: DiceKriging has one stationary kernel per model",
  kern.prod = "kern.prod: DiceKriging has one stationary kernel per model")

# mean.linear_basis (CIP-0009). km() with known covariance parameters fits only the trend:
#   profiled   - generalised-least-squares coefficients; the log likelihood is the concentrated one
#                and predict(type = "UK") adds the trend-uncertainty term r'(M'M)^-1 r.
#   fixed      - coef.trend given; the same expression at those coefficients, predict(type = "SK").
#   flat_prior - the profiled value minus log|A|/2 with A = M'M (M is km's whitened basis).
# gaussian_prior has no counterpart in km, so it is refused.
trend_formula <- function(powers) {
  terms <- apply(powers, 1, function(p) {
    nz <- which(p != 0)
    if (!length(nz)) "I(x1^0)" else paste0("I(", paste0("x", nz, "^", p[nz], collapse = "*"), ")")
  })
  as.formula(paste("~ 0 +", paste(terms, collapse = " + ")))
}

reference <- function(doc) {
  linear <- !is.null(doc$mean) && identical(doc$mean$type, "mean.linear_basis")
  gpscuderia_check_gaussian(doc, names(COVTYPE), WHY_NOT, means = c("mean.constant", "mean.linear_basis"))
  if (linear) {
    mode <- doc$mean$beta$mode
    if (!mode %in% c("fixed", "profiled", "flat_prior"))
      gpscuderia_refuse(sprintf("beta mode '%s': km() estimates the trend by generalised least squares or takes it as given, with no prior on it", mode))
  }
  p <- gpscuderia_gaussian_problem(doc)
  k <- doc$kernel; d <- ncol(p$X)
  # DiceKriging's Matern kernels are products of one-dimensional Matern kernels, which differ
  # from the isotropic wire kernel when d > 1. The Gaussian kernel is the same either way.
  if (d > 1 && k$type != "kern.rbf")
    gpscuderia_refuse(sprintf("%s with %d inputs: DiceKriging's Matern kernel is a product of one-dimensional kernels, not isotropic", k$type, d))
  design <- data.frame(p$X)
  theta <- rep(k$lengthscale, d)          # one shared lengthscale per input dimension
  # noise.var, not nugget: a nugget makes a prediction at a training input noise-free.
  if (linear) {
    names(design) <- paste0("x", seq_len(d))
    args <- list(formula = trend_formula(as.matrix(doc$mean$basis$powers)), design = design,
                 response = p$z, covtype = COVTYPE[[k$type]], coef.cov = theta, coef.var = k$variance,
                 noise.var = rep(p$noise, p$n))
    if (mode == "fixed") args$coef.trend <- as.numeric(doc$mean$beta$value)
    mod <- do.call(km, args)
  } else {
    mod <- km(~1, design = design, response = p$z, covtype = COVTYPE[[k$type]], coef.trend = 0,
              coef.cov = theta, coef.var = k$variance, noise.var = rep(p$noise, p$n))
  }
  # km stores no log likelihood when every parameter is given; it follows from the factor.
  ll <- -0.5 * (p$n * log(2 * pi) + 2 * sum(log(diag(mod@T))) + sum(mod@z^2))
  if (linear && mode == "flat_prior") ll <- ll - sum(log(diag(chol(crossprod(mod@M)))))
  utype <- if (linear && mode != "fixed") "UK" else "SK"
  pr <- predict(mod, setNames(data.frame(p$Xnew), names(design)), type = utype, checkNames = FALSE)
  gpscuderia_gaussian_result(p, ll, pr$mean, pr$sd^2)
}

gpscuderia_main(reference, "dicekriging", gpscuderia_versions(c("DiceKriging", "jsonlite")))
