# Shared driver for Julia runners (CIP-000A), the Julia counterpart of R/gpscuderia/run.R.
#
# A runner script includes this file and calls GPScuderia.main(reference, framework; versions).
# The driver implements the process-level contract: read GPSCUDERIA_TASK, run each golden, write
# GPSCUDERIA_OUT. It passes no judgement on the numbers; scripts/compare.py applies tolerances.
#
#   reference(doc) -> Dict("log_likelihood" => .., "mean" => .., "var" => ..)
#
# `doc` is the golden decoded from its JSON twin (JSON.Object, dictionary-like). Refuse what the
# package cannot express by calling refuse(why) inside reference. Any other error becomes an error
# result for that task only.
module GPScuderia

using JSON, LinearAlgebra, Statistics

struct Unsupported <: Exception
    why::String
end
refuse(why) = throw(Unsupported(why))

# Wall-clock timing as the task contract defines it: one cold call (already made by the caller,
# its duration passed in), `warmup` untimed calls, then `repeats` timed calls.
function timeit(fn, spec, cold)
    for _ in 1:get(spec, "warmup", 0) fn() end
    t = [@elapsed(fn()) for _ in 1:spec["repeats"]]
    q = quantile(t, [0.25, 0.75])
    Dict("seconds_cold" => cold, "seconds_median" => median(t), "seconds_iqr" => q[2] - q[1],
         "repeats" => spec["repeats"])
end

function one(doc, task, reference, gradient)
    ops = get(task, "operations", ["log_likelihood", "predict"])
    if "gradient" in ops
        gradient === nothing && refuse("gradient operation is not provided by this Julia runner")
        any(o -> o in ("log_likelihood", "predict"), ops) || return Dict("gradient" => gradient(doc))
    end
    cold = @elapsed r = reference(doc)
    if "gradient" in ops
        r["gradient"] = gradient(doc)
    end
    haskey(task, "timing") && (r["timing"] = timeit(() -> reference(doc), task["timing"], cold))
    r
end

function main(reference, framework; versions = Dict{String,Any}(), gradient = nothing)
    task = JSON.parsefile(ENV["GPSCUDERIA_TASK"])
    results = Dict{String,Any}()
    for t in task["tasks"]
        results[t["name"]] = try
            one(JSON.parsefile(t["golden"]), t, reference, gradient)
        catch e
            e isa Unsupported ? Dict("refused" => e.why) : Dict("error" => sprint(showerror, e))
        end
    end
    out = Dict{String,Any}("schema" => "gp-scuderia-result/2", "julia" => string(VERSION),
                           "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
                           "results" => results, framework => versions)
    write(ENV["GPSCUDERIA_OUT"], JSON.json(out; allownan = true))
end

# Versions of named packages, for the result file.
function package_versions(pkgs)
    deps = Base.loaded_modules
    Dict{String,Any}(p => string(pkgversion(m)) for (id, m) in deps if id.name in pkgs for p in [id.name])
end

# Matrix of rows from a JSON array of arrays.
rows(a) = reduce(vcat, (permutedims(Float64.(r)) for r in a))

include("gaussian.jl")

end
