# Julia frameworks: AbstractGPs with KernelFunctions (`juliagps`)

The framework runs through the `kind: julia` runner. The harness starts `julia --project=julia` (or the binary named in `JULIA`) on the adapter script, with the task in `GPSCUDERIA_TASK` and the result in `GPSCUDERIA_OUT`. Adapters include `julia/GPScuderia/run.jl`, which implements the process contract and shares `julia/GPScuderia/gaussian.jl`, which refuses what is outside the exact Gaussian slice and converts wire quantities to and from the normalised scale. Packages install into the `julia/` project with `julia/install.jl`, from `ci/pinned-julia.txt` or `ci/latest-julia.txt`. The result records the Julia version, the Julia thread count and the BLAS thread count. The harness sets `JULIA_NUM_THREADS=1` with the other thread variables for single-thread timing.

The first run after installing precompiles for several minutes. A task's `seconds_cold` includes compilation of the model code, so read it as a startup cost, not a model cost.

## Conversions

| Wire | AbstractGPs with KernelFunctions |
|---|---|
| `kern.rbf`, `kern.matern32`, `kern.matern52` | `SqExponentialKernel`, `Matern32Kernel`, `Matern52Kernel` under `with_lengthscale`; no factor-of-two conversion; a vector lengthscale gives ARD |
| `variance` | `variance * kernel` (`ScaledKernel`), one per leaf |
| `kern.bias` | `variance * ConstantKernel()` |
| `kern.white` | `variance * WhiteKernel()`; refused when a test input equals a training input (see below) |
| `kern.add`, `kern.prod` | `+` and `*` |
| Likelihood noise | the noise argument of `GP(k)(x, noise)`, added to the diagonal |
| Log likelihood | `logpdf` of the finite-dimensional GP |
| Predictive variance | `mean_and_var` of the posterior is latent, so the runner adds the noise |
| Mean and normaliser | the runner shifts and scales the targets, then adds the `-n log s` Jacobian |

Probe details are in `julia/probe/NOTES.md`.

## Refusals

- non-exact approximations, other inference methods and non-Gaussian likelihoods;
- `kern.linear` and `kern.coregionalize` (the offset convention of `LinearKernel` is unverified in this round);
- `active_dims`;
- `kern.white` where a test input coincides with a training input. KernelFunctions returns the variance as the cross-covariance there; the wire, like GPy, returns zero.

`scripts/check_julia_refusals.py juliagps` exercises each refusal and the controls that must run.
