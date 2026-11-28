# kergp runner (CIP-0007). Exact GP regression with a fixed set of parameters.
# Run by the harness with GPSCUDERIA_TASK and GPSCUDERIA_OUT set.
root <- Sys.getenv("GPSCUDERIA_ROOT")
source(file.path(root, "R", "gpscuderia", "run.R"))
source(file.path(root, "R", "gpscuderia", "gaussian.R"))
suppressPackageStartupMessages(library(kergp))

KERNELS <- c("kern.rbf", "kern.matern32", "kern.matern52", "kern.bias", "kern.add", "kern.prod")
WHY_NOT <- list(
  kern.white = "kern.white: kergp has no white kernel, and folding it into the noise would change the model's parameters")
FACTOR <- list(kern.rbf = k1Fun1Gauss, kern.matern32 = k1Fun1Matern3_2, kern.matern52 = k1Fun1Matern5_2)

# Distribute the wire tree into a sum of terms; each term is a variance times unit-variance
# radial factors. A product of variances is the product of the factors' variances.
terms <- function(k) {
  switch(k$type,
    kern.rbf = , kern.matern32 = , kern.matern52 = list(list(v = k$variance, f = list(k))),
    kern.bias = list(list(v = k$variance, f = list())),
    kern.add = do.call(c, lapply(k$parts, terms)),
    kern.prod = Reduce(function(A, B) unlist(lapply(A, function(a) lapply(B, function(b)
      list(v = a$v * b$v, f = c(a$f, b$f)))), recursive = FALSE), lapply(k$parts, terms)))
}

build_kernel <- function(k, inp) {
  d <- length(inp)
  env <- new.env()
  const <- function(v) covMan(kernel = function(x1, x2, par) par[1],
                              d = d, inputs = inp, parNames = "v", par = v, label = "const")
  parts <- character()
  for (i in seq_along(terms(k))) {
    tm <- terms(k)[[i]]
    assign(paste0("c", i), const(tm$v), env)
    fs <- character()
    for (j in seq_along(tm$f)) {
      args <- list(k1Fun1 = FACTOR[[tm$f[[j]]$type]], cov = "corr", d = d, inputs = inp, par = tm$f[[j]]$lengthscale)
      if (d > 1) args$iso <- 1
      assign(paste0("f", i, "_", j), do.call(covRadial, args), env)
      fs <- c(fs, paste0("f", i, "_", j, "()"))
    }
    parts <- c(parts, paste(c(paste0("c", i, "()"), fs), collapse = " * "))
  }
  covComp(formula = as.formula(paste("~", paste(parts, collapse = " + ")), env = env), where = env)
}

reference <- function(doc) {
  gpscuderia_check_gaussian(doc, KERNELS, WHY_NOT)
  p <- gpscuderia_gaussian_problem(doc)
  inp <- paste0("x", seq_len(ncol(p$X)))
  colnames(p$X) <- inp; colnames(p$Xnew) <- inp
  fit <- gp(z ~ 1, data = data.frame(p$X, z = p$z), inputs = inp, cov = build_kernel(doc$kernel, inp),
            estim = FALSE, varNoise = p$noise, beta = 0)
  # gp(estim = FALSE) leaves logLik NA; the Gaussian log likelihood follows from its factor.
  ll <- -0.5 * (p$n * log(2 * pi) + 2 * sum(log(diag(fit$L))) + as.numeric(fit$sseStar))
  pr <- predict(fit, data.frame(p$Xnew), type = "SK", checkNames = FALSE)   # forceInterp stays FALSE
  gpscuderia_gaussian_result(p, ll, pr$mean, pr$sd^2)
}

gpscuderia_main(reference, "kergp", gpscuderia_versions(c("kergp", "jsonlite")))
