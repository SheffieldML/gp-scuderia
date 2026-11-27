# Independent reference for the wire semantics (CIP-0007 phase 0 probe).
#
# This is deliberately naive dense R: it exists to (a) check our reading of the
# wire format against the GPy-generated x-reference values and (b) give an
# elementwise covariance target against which each R package is compared.
# It is not a runner and is not used for conformance.

suppressPackageStartupMessages(library(jsonlite))

read_golden <- function(name, dir = "examples") {
  fromJSON(file.path(dir, paste0(name, ".json")), simplifyVector = FALSE)
}

as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))

sqdist <- function(X1, X2, ell) {
  # Scaled squared distances; ell is length 1 or ncol(X).
  ell <- rep_len(ell, ncol(X1))
  A <- sweep(X1, 2, ell, "/")
  B <- sweep(X2, 2, ell, "/")
  outer(rowSums(A^2), rowSums(B^2), "+") - 2 * A %*% t(B)
}

wire_kernel <- function(k, X1, X2 = X1) {
  same <- identical(X1, X2)
  switch(k$type,
    "kern.rbf" = {
      r2 <- pmax(sqdist(X1, X2, unlist(k$lengthscale)), 0)
      k$variance * exp(-0.5 * r2)
    },
    "kern.matern32" = {
      r <- sqrt(pmax(sqdist(X1, X2, unlist(k$lengthscale)), 0))
      k$variance * (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
    },
    "kern.matern52" = {
      r <- sqrt(pmax(sqdist(X1, X2, unlist(k$lengthscale)), 0))
      k$variance * (1 + sqrt(5) * r + 5 / 3 * r^2) * exp(-sqrt(5) * r)
    },
    "kern.bias" = matrix(k$variance, nrow(X1), nrow(X2)),
    "kern.white" = if (same) k$variance * diag(nrow(X1)) else matrix(0, nrow(X1), nrow(X2)),
    "kern.add" = Reduce(`+`, lapply(k$parts, wire_kernel, X1 = X1, X2 = X2)),
    "kern.prod" = Reduce(`*`, lapply(k$parts, wire_kernel, X1 = X1, X2 = X2)),
    stop("probe wire reference: unsupported kernel ", k$type)
  )
}

# Zero-mean GP with Gram = K + noise I, following docs/overview.md.
# Normalizer: model fitted to (y - m) / s; reported on the original scale with
# the -n log s Jacobian. Mean: prior mean c on the original scale.
wire_reference <- function(d) {
  X <- as_matrix(d$data$X); y <- as_matrix(d$data$Y)[, 1]
  Xn <- as_matrix(d$`x-reference`$Xnew)
  m <- 0; s <- 1
  if (!is.null(d$normalizer)) {
    m <- d$normalizer$mean
    s <- if (is.null(d$normalizer$std)) 1 else d$normalizer$std
  }
  if (!is.null(d$mean)) m <- d$mean$value
  z <- (y - m) / s
  n <- length(z)
  K <- wire_kernel(d$kernel, X) + d$likelihood$variance * diag(n)
  L <- chol(K)
  a <- backsolve(L, forwardsolve(t(L), z))
  ll <- -0.5 * sum(z * a) - sum(log(diag(L))) - 0.5 * n * log(2 * pi) - n * log(s)
  Ks <- wire_kernel(d$kernel, Xn, X)
  Kss <- diag(wire_kernel(d$kernel, Xn, Xn))
  v <- forwardsolve(t(L), t(Ks))
  list(
    log_likelihood = ll,
    mean = as.numeric(Ks %*% a) * s + m,
    var_latent = (Kss - colSums(v^2)) * s^2,
    noise = d$likelihood$variance * s^2
  )
}

check_reference <- function(name, dir = "examples") {
  d <- read_golden(name, dir)
  r <- wire_reference(d); ref <- d$`x-reference`
  data.frame(
    golden = name,
    d_loglik = r$log_likelihood - ref$log_likelihood,
    d_mean = max(abs(r$mean - unlist(ref$mean))),
    d_var_latent = max(abs(r$var_latent - unlist(ref$var))),
    d_var_noisy = max(abs(r$var_latent + r$noise - unlist(ref$var)))
  )
}

if (sys.nframe() == 0) {
  goldens <- c("ftc_rbf_white", "ftc_standardize", "ftc_prod_matern52")
  print(do.call(rbind, lapply(goldens, check_reference)), digits = 3)
}
