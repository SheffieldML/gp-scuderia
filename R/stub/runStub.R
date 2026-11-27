# Stub R runner for scripts/check_runner_contract.py (CIP-0007).
# Returns each golden's own reference, so it needs no GP package. Behaviour is selected by the
# golden's reference log-likelihood (unique per golden): crash the process, raise, or refuse.
# GPSCUDERIA_STUB_CRASH, _RAISE and _REFUSE carry those log-likelihoods.
root <- Sys.getenv("GPSCUDERIA_ROOT")
source(file.path(root, "R", "gpscuderia", "run.R"))

hit <- function(env, ll) abs(ll - as.numeric(Sys.getenv(env))) < 1e-9

reference <- function(doc) {
  ref <- doc[["x-reference"]]
  ll <- ref$log_likelihood
  if (hit("GPSCUDERIA_STUB_CRASH", ll)) quit(save = "no", status = 3)
  if (hit("GPSCUDERIA_STUB_RAISE", ll)) stop("stub failure")
  if (hit("GPSCUDERIA_STUB_REFUSE", ll)) gpscuderia_refuse("stub refusal")
  list(log_likelihood = ll, mean = ref$mean, var = ref$var)
}

gpscuderia_main(reference, "stub", gpscuderia_versions(c("jsonlite")))
