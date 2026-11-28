# Shared driver for R runners (CIP-0007), the R counterpart of octave/gpscuderia.
#
# A runner script sources this file and calls gpscuderia_main(reference, ...). The driver
# implements the process-level contract: read GPSCUDERIA_TASK, run each golden, write
# GPSCUDERIA_OUT. It passes no judgement on the numbers; scripts/compare.py applies tolerances.
#
#   reference(doc) -> list(log_likelihood, mean, var, [latent_mean, latent_var])
#
# `doc` is the golden as decoded from its JSON twin. Matrices (data$X, data$Y, Xnew) arrive
# as numeric matrices. Beware that jsonlite collapses a length-one array to a scalar, so a
# one-dimensional ARD lengthscale arrives as a number; wrap such fields in as.numeric().
#
# Refuse what the package cannot express by calling gpscuderia_refuse(why) inside reference.
# Any other error becomes an error result for that task only.

suppressPackageStartupMessages(library(jsonlite))

gpscuderia_refuse <- function(why) {
  stop(structure(class = c("gpscuderia_unsupported", "error", "condition"),
                 list(message = why, call = NULL)))
}

gpscuderia_read <- function(path) {
  fromJSON(path, simplifyVector = TRUE, simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
}

# Wall-clock timing as the task contract defines it: one cold call (already made by the
# caller, its duration passed in), `warmup` untimed calls, then `repeats` timed calls.
gpscuderia_time <- function(fn, spec, cold) {
  warmup <- if (is.null(spec$warmup)) 0L else spec$warmup
  for (i in seq_len(warmup)) fn()
  t <- vapply(seq_len(spec$repeats), function(i) system.time(fn())[["elapsed"]], numeric(1))
  list(seconds_cold = cold, seconds_median = median(t), seconds_iqr = IQR(t), repeats = spec$repeats)
}

gpscuderia_one <- function(doc, task, reference, gradient) {
  ops <- unlist(task$operations)
  if (is.null(ops)) ops <- c("log_likelihood", "predict")
  if ("gradient" %in% ops) {
    if (is.null(gradient)) gpscuderia_refuse("gradient operation is not provided by this R runner")
    if (!any(c("log_likelihood", "predict") %in% ops)) return(list(gradient = gradient(doc)))
  }
  cold <- system.time(r <- reference(doc))[["elapsed"]]
  for (k in c("mean", "var", "latent_mean", "latent_var")) if (!is.null(r[[k]])) r[[k]] <- I(as.numeric(r[[k]]))
  if ("gradient" %in% ops) r$gradient <- gradient(doc)
  if (!is.null(task$timing)) r$timing <- gpscuderia_time(function() reference(doc), task$timing, cold)
  r
}

gpscuderia_main <- function(reference, framework, versions = character(), gradient = NULL) {
  warnings_off <- options(warn = -1); on.exit(options(warnings_off))
  task <- fromJSON(Sys.getenv("GPSCUDERIA_TASK"), simplifyVector = FALSE)
  results <- list()
  for (t in task$tasks) {
    results[[t$name]] <- tryCatch(
      gpscuderia_one(gpscuderia_read(t$golden), t, reference, gradient),
      gpscuderia_unsupported = function(e) list(refused = conditionMessage(e)),
      error = function(e) list(error = conditionMessage(e)))
  }
  out <- list(schema = "gp-scuderia-result/2", r = R.version.string, results = results)
  out[[framework]] <- as.list(versions)
  write_json(out, Sys.getenv("GPSCUDERIA_OUT"), auto_unbox = TRUE, digits = NA, null = "null")
}

# Versions of named packages, for the result file.
gpscuderia_versions <- function(pkgs) {
  setNames(lapply(pkgs, function(p) tryCatch(as.character(packageVersion(p)), error = function(e) NA)), pkgs)
}
