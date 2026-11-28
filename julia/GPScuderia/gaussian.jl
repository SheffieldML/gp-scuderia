# Wire-document handling shared by the Julia exact-GP runners (CIP-000A).
#
# Wire semantics (docs/overview.md): Gram = K + noise I; kernel and noise parameters live on the
# normalised scale; the reported log likelihood includes the -n log s Jacobian; the predictive
# variance includes the noise.

function check_gaussian(doc, kernels, why_not = Dict{String,String}())
    approx = get(doc, "approximation", "ftc")
    approx == "ftc" || refuse("approximation '$approx' is not realised by this package")
    lik = doc["likelihood"]["type"]
    lik == "likelihoods.gaussian" || refuse("$lik: the package fits Gaussian noise only")
    inf = get(doc, "inference", "exact")
    inf == "exact" || refuse("inference '$inf' is not realised by this package")
    haskey(doc, "mean") && haskey(doc, "normalizer") && refuse("mean and normalizer together are not in schema 0.1")
    function walk(k)
        t = k["type"]
        t in kernels || refuse(get(why_not, t, "$t is not realised by this package"))
        haskey(k, "active_dims") && refuse("active_dims is not yet supported by this runner")
        foreach(walk, get(k, "parts", []))
    end
    walk(doc["kernel"])
end

# Training data, the target transform, and the noise, ready for a package.
function gaussian_problem(doc)
    X = rows(doc["data"]["X"])
    y = Float64[r[1] for r in doc["data"]["Y"]]
    Xn = rows(doc["x-reference"]["Xnew"])
    m, s = 0.0, 1.0
    if haskey(doc, "normalizer")
        m = Float64(doc["normalizer"]["mean"]); s = Float64(get(doc["normalizer"], "std", 1.0))
    end
    haskey(doc, "mean") && (m = Float64(doc["mean"]["value"]))
    (X = X, Xnew = Xn, z = (y .- m) ./ s, shift = m, scale = s, n = length(y),
     noise = Float64(doc["likelihood"]["variance"]), jacobian = length(y) * log(s))
end

# Map the package's results on the normalised scale back to the wire. `var_latent` is the latent
# predictive variance (no noise): the wire reports the variance of a new observation.
gaussian_result(p, loglik, mean, var_latent) = Dict{String,Any}(
    "log_likelihood" => loglik - p.jacobian,
    "mean" => collect(mean .* p.scale .+ p.shift),
    "var" => collect((var_latent .+ p.noise) .* p.scale^2))
