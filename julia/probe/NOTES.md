# CIP-000A phase 0 probe notes

Environment: Julia 1.11.7 (macOS arm64), AbstractGPs 0.5.25, KernelFunctions 0.11.1,
GaussianProcesses 0.12.6, JSON 1.13.4. All parameters fixed; no optimisation.

## AbstractGPs with KernelFunctions

Matches `ftc_rbf_white` (loglik 2.6e-8, mean 1e-9, variance 5e-9 against the GPy reference) and
`ftc_prod_matern52` (`add(prod(matern52, rbf), bias)`; loglik 2e-8, mean 1e-9, variance 1e-9).
An ARD RBF matches a dense calculation of the log likelihood to 7e-14.

- Kernels: `SqExponentialKernel`, `Matern32Kernel`, `Matern52Kernel` with `with_lengthscale(k, l)`
  equal `kern.rbf`, `kern.matern32`, `kern.matern52`. No factor-of-two conversion. A vector
  lengthscale gives ARD. `variance * k` is `ScaledKernel`; `ConstantKernel()` scaled gives `kern.bias`.
  `+` and `*` give `kern.add` and `kern.prod`. Each leaf carries its own variance, as on the wire.
- Noise: `GP(k)(x, noise)` adds `noise` to the diagonal. `logpdf` is the full Gaussian value.
  `mean_and_var(posterior(fx, y)(xnew))` is the latent variance; the runner adds the noise.
- White: `WhiteKernel` equals the wire at inputs distinct from the training set. At an input that
  equals a training input it returns the variance as cross-covariance, where the wire (and GPy)
  return zero. The runner refuses a golden whose `Xnew` coincides with a training row and has a
  white kernel, rather than return a different number.
- Mean and target scale: handled by the runner shifting and scaling the data, so the zero-mean GP
  is exact. The `-n log s` Jacobian is added to the log likelihood.
- JSON: `JSON.parsefile` returns a `JSON.Object` (dictionary-like). Arrays arrive as `Vector{Any}`,
  and a length-one array stays an array, so ARD lengthscales need no scalar-collapse guard.
- Not checked: gradients, `kern.linear` (`LinearKernel` has an offset convention to verify),
  active dimensions, small-noise conditioning (AbstractGPs adds no jitter of its own).

## GaussianProcesses.jl

Matches AbstractGPs to 1e-15 on an RBF with variance 2 and on an ARD RBF.

- Parameters are logs: `SE(log l, log sigma)` where `sigma` is the standard deviation, so the wire
  variance `v` maps to `log(sqrt(v))`. `SEArd` takes a vector of log lengthscales. `Mat32Iso`,
  `Mat52Iso`, `Mat32Ard`, `Mat52Ard` follow the same pattern. `Const(log sigma)` carries a constant
  covariance `sigma^2` (`kern.bias`). The noise argument of `GPE` is `log` of the noise standard deviation.
- There is no white kernel. The noise lives in the `GPE`, as in GPyTorch, so `kern.white` is refused.
- `gp.target` is the log likelihood. `predict_f` is latent, so the runner adds the noise.
- `MeanConst(c)` carries `mean.constant`; the runner shift makes it unnecessary.
- Inputs are `d x n` (the transpose of the wire).
- Not checked: `+` and `*` of kernels, other kernels in the package (`Lin`, `RQ`).

## Consequences for the CIP

- `juliagps` kernel list keeps rbf, matern32, matern52 (all with ARD), bias, white, add, prod.
  `linear` is dropped from the first round until its convention is verified.
- `gaussianprocesses` loses `white` from the expected list; add `bias` (`Const`).
- Startup: loading AbstractGPs and KernelFunctions is under 10 s warm, but the first run after
  `Pkg.instantiate` precompiles for several minutes. The cold time in a result includes
  compilation of the model code, so benchmarks must report `seconds_cold` separately.
