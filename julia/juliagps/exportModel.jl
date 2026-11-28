# Write a wire document for an AbstractGPs model built natively in Julia (CIP-000A round trip).
# Usage: julia --project=julia julia/juliagps/exportModel.jl OUT.json
# The model is a Gaussian GP regression with an RBF kernel; the x-reference block is computed by
# AbstractGPs itself, so GPy reading this file is an independent check of the export.
using AbstractGPs, KernelFunctions, JSON

variance, lengthscale, noise = 1.3, 0.7, 0.05
X = collect(range(-2.0, 2.5; length=7))
y = sin.(2 .* X) .+ 0.1 .* cos.(7 .* X)
Xnew = [-1.5, 0.5, 3.0]

k = variance * with_lengthscale(SqExponentialKernel(), lengthscale)
fx = AbstractGPs.GP(k)(RowVecs(reshape(X, :, 1)), noise)
m, v = mean_and_var(posterior(fx, y)(RowVecs(reshape(Xnew, :, 1))))

doc = Dict(
    "schema_version" => "0.1", "type" => "models.gp_regression", "input_dim" => 1, "output_dim" => 1,
    "approximation" => "ftc",
    "kernel" => Dict("type" => "kern.rbf", "variance" => variance, "lengthscale" => lengthscale, "ARD" => false),
    "likelihood" => Dict("type" => "likelihoods.gaussian", "variance" => noise),
    "data" => Dict("X" => [[x] for x in X], "Y" => [[t] for t in y]),
    "x-reference" => Dict("oracle" => "AbstractGPs", "Xnew" => [[x] for x in Xnew],
        "log_likelihood" => logpdf(fx, y), "mean" => m, "var" => v .+ noise,
        "tolerance" => Dict("log_likelihood" => 1e-8, "mean" => 1e-8, "var" => 1e-8)),
)
write(ARGS[1], JSON.json(doc))
