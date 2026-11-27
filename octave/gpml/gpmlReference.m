function out = gpmlReference(doc, mode)
% Fit nothing, optimise nothing: evaluate the wire model with GPML and predict at x-reference.Xnew.
% Returns a struct with log_likelihood, mean, var (and latent_mean, latent_var for non-Gaussian).
% Raises gpscuderia:unsupported for anything GPML does not realise.
X = doc.data.X; Y = doc.data.Y; Xs = doc.x_reference.Xnew;
if size(Y, 2) ~= 1, error('gpscuderia:unsupported', 'multi-output is not realised by GPML'); end
if ~isfield(doc, 'inference'), inference = 'none'; else inference = doc.inference; end
[cov, hyp.cov] = gpmlBuildKernel(doc.kernel);
meanf = {@meanZero}; hyp.mean = [];
if isfield(doc, 'mean')
  if ~strcmp(doc.mean.type, 'mean.constant'), error('gpscuderia:unsupported', '%s mean', doc.mean.type); end
  meanf = {@meanConst}; hyp.mean = doc.mean.value;
end
ltype = doc.likelihood.type;
scale = 1; shift = 0; post = @(m, v) deal(m, v);
switch ltype
  case 'likelihoods.gaussian'
    if ~strcmp(inference, 'none'), error('gpscuderia:unsupported', 'gaussian likelihood with %s inference', inference); end
    lik = {@likGauss}; hyp.lik = log(sqrt(doc.likelihood.variance));
    inf = @infGaussLik;
  case 'likelihoods.bernoulli'
    if ~strcmp(inference, 'laplace'), error('gpscuderia:unsupported', 'bernoulli needs laplace inference'); end
    lik = {@likErf}; hyp.lik = []; inf = @infLaplace; Y = 2 * Y - 1;
  case 'likelihoods.poisson'
    if ~strcmp(inference, 'laplace'), error('gpscuderia:unsupported', 'poisson needs laplace inference'); end
    lik = {@likPoisson, 'exp'}; hyp.lik = []; inf = @infLaplace;
  otherwise
    error('gpscuderia:unsupported', '%s is not realised by GPML', ltype);
end
if isfield(doc, 'normalizer')
  if ~strcmp(doc.normalizer.type, 'normalizer.standardize'), error('gpscuderia:unsupported', '%s', doc.normalizer.type); end
  shift = doc.normalizer.mean; scale = doc.normalizer.std; Y = (Y - shift) / scale;
end
if isfield(doc, 'mean') && isfield(doc, 'normalizer'), error('gpscuderia:unsupported', 'mean and normalizer together'); end
switch doc.approximation
  case 'ftc'
  case 'fitc'
    if ~strcmp(ltype, 'likelihoods.gaussian'), error('gpscuderia:unsupported', 'fitc with %s', ltype); end
    cov = {@covFITC, cov, doc.inducing_inputs}; inf = @infFITC;
  otherwise
    error('gpscuderia:unsupported', 'approximation ''%s'' is not realised by GPML', doc.approximation);
end
if strcmp(ltype, 'likelihoods.gaussian') && strcmp(doc.approximation, 'ftc'), inf = @infGaussLik; end
if nargin > 1 && strcmp(mode, 'gradient')
  if ~strcmp(ltype, 'likelihoods.gaussian'), error('gpscuderia:unsupported', 'gradient: %s is outside the gradient contract', ltype); end
  [~, dnlZ] = gp(hyp, inf, meanf, cov, lik, X, Y);
  out = struct('gradient', gpmlGradients(doc, dnlZ));
  return
end
[nlZ] = gp(hyp, inf, meanf, cov, lik, X, Y);
[ymu, ys2, fmu, fs2] = gp(hyp, inf, meanf, cov, lik, X, Y, Xs);
out.log_likelihood = -nlZ - size(Y, 1) * log(scale);  % golden is in original units (Jacobian of the standardisation)
switch ltype
  case 'likelihoods.gaussian'
    out.mean = ymu * scale + shift; out.var = ys2 * scale^2;
  case 'likelihoods.bernoulli'
    p = (ymu + 1) / 2; out.mean = p; out.var = p .* (1 - p);
    out.latent_mean = fmu; out.latent_var = fs2;
  case 'likelihoods.poisson'
    out.mean = ymu; out.var = ys2; out.latent_mean = fmu; out.latent_var = fs2;
end
out.mean = out.mean(:)'; out.var = out.var(:)';
if isfield(out, 'latent_mean'), out.latent_mean = out.latent_mean(:)'; out.latent_var = out.latent_var(:)'; end
end

function g = gpmlGradients(doc, dnlZ)
% GPML differentiates nlZ with respect to log hyperparameters: log lengthscale, log sf and
% log sn with variance = sf^2 = sn^2. Convert to the wire parameters (variance, lengthscale):
%   dLL/dls = -dnlZ/ls     dLL/dvariance = -dnlZ/(2 variance)
g = containers.Map();
g = leafGradients(g, doc.kernel, 'kernel', dnlZ.cov, 0);
g('likelihood.variance') = -dnlZ.lik(1) / (2 * doc.likelihood.variance);
end

function [g, pos] = leafGradients(g, k, path, d, pos)
if isfield(k, 'parts')
  parts = k.parts; if isstruct(parts), parts = num2cell(parts); end
  for i = 1:numel(parts)
    [g, pos] = leafGradients(g, parts{i}, sprintf('%s.parts.%d', path, i - 1), d, pos);
  end
  return
end
switch k.type
  case {'kern.rbf', 'kern.matern32', 'kern.matern52'}
    n = numel(k.lengthscale); ls = k.lengthscale(:)';
    g([path '.lengthscale']) = -d(pos + (1:n))(:)' ./ ls; pos = pos + n;
    g([path '.variance']) = -d(pos + 1) / (2 * k.variance); pos = pos + 1;
  case {'kern.bias', 'kern.white'}
    g([path '.variance']) = -d(pos + 1) / (2 * k.variance); pos = pos + 1;
  otherwise
    error('gpscuderia:unsupported', 'gradient: %s', k.type);
end
end
