# Stub Julia runner for scripts/check_runner_contract.py (CIP-000A).
# Returns each golden's own reference, so it needs no GP package. Behaviour is selected by the
# golden's reference log-likelihood (unique per golden): crash the process, raise, or refuse.
# GPSCUDERIA_STUB_CRASH, _RAISE and _REFUSE carry those log-likelihoods.
include(joinpath(ENV["GPSCUDERIA_ROOT"], "julia", "GPScuderia", "run.jl"))
using .GPScuderia: refuse

hit(env, ll) = abs(ll - parse(Float64, ENV[env])) < 1e-9

function reference(doc)
    ref = doc["x-reference"]
    ll = ref["log_likelihood"]
    hit("GPSCUDERIA_STUB_CRASH", ll) && exit(3)
    hit("GPSCUDERIA_STUB_RAISE", ll) && error("stub failure")
    hit("GPSCUDERIA_STUB_REFUSE", ll) && refuse("stub refusal")
    Dict("log_likelihood" => ll, "mean" => collect(ref["mean"]), "var" => collect(ref["var"]))
end

GPScuderia.main(reference, "stub", versions = GPScuderia.package_versions(["JSON"]))
