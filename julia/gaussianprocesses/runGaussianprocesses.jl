# GaussianProcesses.jl runner (CIP-000A, phase 2). Exact GP regression with fixed parameters.
# Run by the harness with GPSCUDERIA_TASK and GPSCUDERIA_OUT set.
include(joinpath(ENV["GPSCUDERIA_ROOT"], "julia", "GPScuderia", "run.jl"))
import GaussianProcesses as GPs
using .GPScuderia: refuse

# No white kernel: GaussianProcesses.jl has none (probe in julia/probe/NOTES.md), so kern.white is
# refused by omission rather than approximated with the noise term.
const KERNELS = ["kern.rbf", "kern.matern32", "kern.matern52", "kern.bias", "kern.add", "kern.prod"]

# GaussianProcesses.jl stores log lengthscale and log standard deviation: sigma = sqrt(variance).
# Checked against the wire Gram: SE(log l, log sigma) has the wire's lengthscale convention.
function build_kernel(k)
    t = k["type"]
    t == "kern.add" && return reduce(+, map(build_kernel, k["parts"]))
    t == "kern.prod" && return reduce(*, map(build_kernel, k["parts"]))
    ls = 0.5 * log(Float64(k["variance"]))
    t == "kern.bias" && return GPs.Const(ls)
    l = k["lengthscale"]
    ard = l isa AbstractVector
    ll = ard ? log.(Float64.(l)) : log(Float64(l))
    if t == "kern.rbf"
        return ard ? GPs.SEArd(ll, ls) : GPs.SE(ll, ls)
    elseif t == "kern.matern32"
        return ard ? GPs.Mat32Ard(ll, ls) : GPs.Mat32Iso(ll, ls)
    else
        return ard ? GPs.Mat52Ard(ll, ls) : GPs.Mat52Iso(ll, ls)
    end
end

function reference(doc)
    GPScuderia.check_gaussian(doc, KERNELS)
    p = GPScuderia.gaussian_problem(doc)
    # Inputs are d x n.
    gp = GPs.GPE(permutedims(p.X), Float64.(p.z), GPs.MeanZero(), build_kernel(doc["kernel"]), 0.5 * log(p.noise))
    m, v = GPs.predict_f(gp, permutedims(p.Xnew))
    GPScuderia.gaussian_result(p, gp.target, m, v)
end

GPScuderia.main(reference, "gaussianprocesses",
               versions = GPScuderia.package_versions(["GaussianProcesses", "JSON"]))
