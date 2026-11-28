# Install the Julia packages the Julia runners need (CIP-000A).
#   julia --project=julia julia/install.jl ci/pinned-julia.txt   # exact versions, "package version" per line
#   julia --project=julia julia/install.jl ci/latest-julia.txt   # newest release, "package" per line
using Pkg
specs = PackageSpec[]
for l in strip.(readlines(only(ARGS)))
    (isempty(l) || startswith(l, "#")) && continue
    f = split(l)
    push!(specs, length(f) > 1 ? PackageSpec(name = f[1], version = f[2]) : PackageSpec(name = f[1]))
end
Pkg.add(specs)
all(s -> s.version === nothing || s.version == Pkg.Types.VersionSpec(), specs) && Pkg.update()   # latest: newest releases, not the committed Manifest
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
