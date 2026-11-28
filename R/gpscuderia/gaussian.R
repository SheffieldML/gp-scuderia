# Wire-document handling shared by the R exact-GP runners (CIP-0007).
#
# Both packages fit a zero-mean (or known-mean) Gaussian process with Gaussian noise by exact
# inference. This file validates the golden against that slice, refuses everything else, and
# maps the wire's target transform and noise onto the form the packages need. Parameter
# conventions that differ per package (lengthscale, nugget) live in the runners.
#
# Wire semantics (docs/overview.md): Gram = K + noise I; kernel and noise parameters live on
# the normalised scale; the reported log likelihood includes the -n log s Jacobian; the
# predictive variance includes the noise.

gpscuderia_check_gaussian <- function(doc, kernels, why_not = list(), means = "mean.constant") {
  if (!identical(if (is.null(doc$approximation)) "ftc" else doc$approximation, "ftc"))
    gpscuderia_refuse(sprintf("approximation '%s' is not realised by this package", doc$approximation))
  if (!identical(doc$likelihood$type, "likelihoods.gaussian"))
    gpscuderia_refuse(sprintf("%s: the package fits Gaussian noise only", doc$likelihood$type))
  if (!identical(if (is.null(doc$inference)) "exact" else doc$inference, "exact"))
    gpscuderia_refuse(sprintf("inference '%s' is not realised by this package", doc$inference))
  if (!is.null(doc$mean) && !doc$mean$type %in% means)
    gpscuderia_refuse(sprintf("%s is not realised by this package", doc$mean$type))
  if (!is.null(doc$mean) && !is.null(doc$normalizer))
    gpscuderia_refuse("mean and normalizer together are not in schema 0.1")
  walk <- function(k) {
    if (!k$type %in% kernels) {
      why <- why_not[[k$type]]
      gpscuderia_refuse(if (is.null(why)) sprintf("%s is not realised by this package", k$type) else why)
    }
    if (isTRUE(k$ARD) || length(k$lengthscale) > 1)
      gpscuderia_refuse(sprintf("%s with ARD is not yet supported by this runner", k$type))
    if (!is.null(k$active_dims))
      gpscuderia_refuse("active_dims is not yet supported by this runner")
    for (p in k$parts) walk(p)
  }
  walk(doc$kernel)
}

# Training data, the target transform, and the noise, ready for a package.
gpscuderia_gaussian_problem <- function(doc) {
  X <- as.matrix(doc$data$X)
  y <- as.numeric(doc$data$Y)
  Xn <- as.matrix(doc[["x-reference"]]$Xnew)
  m <- 0; s <- 1
  if (!is.null(doc$normalizer)) {
    m <- doc$normalizer$mean
    s <- if (is.null(doc$normalizer$std)) 1 else doc$normalizer$std
  }
  if (!is.null(doc$mean) && identical(doc$mean$type, "mean.constant")) m <- doc$mean$value
  list(X = X, Xnew = Xn, z = (y - m) / s, shift = m, scale = s, n = length(y),
       noise = doc$likelihood$variance, jacobian = length(y) * log(s))
}

# Map the package's results on the normalised scale back to the wire. `var_latent` is the
# latent predictive variance (no noise): the wire reports the variance of a new observation.
gpscuderia_gaussian_result <- function(p, loglik, mean, var_latent) {
  list(log_likelihood = loglik - p$jacobian,
       mean = mean * p$scale + p$shift,
       var = (var_latent + p$noise) * p$scale^2)
}
