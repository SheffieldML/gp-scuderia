# GauPro runner (CIP-0007). Exact GP regression with fixed parameters.
# Run by the harness with GPSCUDERIA_TASK and GPSCUDERIA_OUT set.
root <- Sys.getenv("GPSCUDERIA_ROOT")
source(file.path(root, "R", "gpscuderia", "run.R"))
source(file.path(root, "R", "gpscuderia", "gaussian.R"))
suppressPackageStartupMessages(library(GauPro))

KERNELS <- c("kern.rbf", "kern.matern32", "kern.matern52", "kern.add", "kern.prod")
WHY_NOT <- list(kern.bias = "kern.bias: GauPro has no constant kernel",
                kern.white = "kern.white: GauPro White kernel fails when its parameters are fixed (param_optim_start)")

# GauPro takes theta = 10^beta per input dimension. Verified against the wire Gram to 1e-16:
# Gaussian is exp(-theta r^2) with theta = 1 / (2 l^2); the Matern kernels use theta = 1 / l^2.
beta_of <- function(type, l) if (type == "kern.rbf") -log10(2 * l^2) else -2 * log10(l)
CLASS <- list(kern.rbf = Gaussian, kern.matern32 = Matern32, kern.matern52 = Matern52)

build_kernel <- function(k, d) {
  switch(k$type,
    kern.add = Reduce(`+`, lapply(k$parts, build_kernel, d = d)),
    kern.prod = Reduce(`*`, lapply(k$parts, build_kernel, d = d)),
    CLASS[[k$type]]$new(D = d, beta = rep(beta_of(k$type, k$lengthscale), d), s2 = k$variance,
                        s2_est = FALSE, beta_est = FALSE))
}

reference <- function(doc) {
  gpscuderia_check_gaussian(doc, KERNELS, WHY_NOT)
  p <- gpscuderia_gaussian_problem(doc)
  kern <- build_kernel(doc$kernel, ncol(p$X))
  # The nugget is relative to the kernel's total variance: K = s2 (R + nug I). A sum or product
  # of kernels carries its own s2 (the sum or product of the parts' variances).
  mod <- GauPro_kernel_model$new(X = p$X, Z = p$z, kernel = kern, trend = trend_0$new(),
                                 nug = p$noise / kern$s2, nug.est = FALSE, nug.min = 1e-12,
                                 param.est = FALSE, normalize = FALSE)
  pr <- mod$predict(p$Xnew, se.fit = TRUE)
  # GauPro's predictive variance already includes the noise; the result helper adds it back.
  gpscuderia_gaussian_result(p, mod$loglikelihood(), pr$mean, pr$s2 - p$noise)
}

gpscuderia_main(reference, "gaupro", gpscuderia_versions(c("GauPro", "jsonlite")))
