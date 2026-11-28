# AbstractGPs + KernelFunctions runner (CIP-000A). Exact GP regression with fixed parameters.
# Run by the harness with GPSCUDERIA_TASK and GPSCUDERIA_OUT set.
include(joinpath(ENV["GPSCUDERIA_ROOT"], "julia", "GPScuderia", "run.jl"))
using AbstractGPs, KernelFunctions
using .GPScuderia: refuse

const KERNELS = ["kern.rbf", "kern.matern32", "kern.matern52", "kern.bias", "kern.white", "kern.add", "kern.prod"]
const BASE = Dict("kern.rbf" => SqExponentialKernel(), "kern.matern32" => Matern32Kernel(),
                  "kern.matern52" => Matern52Kernel())

# Verified against the wire Gram: with_lengthscale(k, l) has the wire's convention with no factor
# of two, and a vector lengthscale is ARD. Each leaf carries its own variance.
function build_kernel(k)
    t = k["type"]
    t == "kern.add" && return reduce(+, map(build_kernel, k["parts"]))
    t == "kern.prod" && return reduce(*, map(build_kernel, k["parts"]))
    v = Float64(k["variance"])
    t == "kern.bias" && return v * ConstantKernel()
    t == "kern.white" && return v * WhiteKernel()
    l = k["lengthscale"]
    l = l isa Number ? Float64(l) : Float64.(l)
    v * with_lengthscale(BASE[t], l)
end

haswhite(k) = k["type"] == "kern.white" || any(haswhite, get(k, "parts", []))

function reference(doc)
    GPScuderia.check_gaussian(doc, KERNELS)
    p = GPScuderia.gaussian_problem(doc)
    # KernelFunctions' WhiteKernel is one at coinciding inputs, the wire (and GPy) is zero in the
    # cross-covariance. Differ only where a test input repeats a training input: refuse there.
    if haswhite(doc["kernel"]) && any(i -> any(j -> p.Xnew[i, :] == p.X[j, :], axes(p.X, 1)), axes(p.Xnew, 1))
        refuse("kern.white at a test input that equals a training input: KernelFunctions returns the variance as cross-covariance, the wire returns zero")
    end
    fx = AbstractGPs.GP(build_kernel(doc["kernel"]))(RowVecs(p.X), p.noise)
    m, v = mean_and_var(posterior(fx, p.z)(RowVecs(p.Xnew)))
    GPScuderia.gaussian_result(p, logpdf(fx, p.z), m, v)
end

GPScuderia.main(reference, "juliagps",
               versions = GPScuderia.package_versions(["AbstractGPs", "KernelFunctions", "JSON"]))
