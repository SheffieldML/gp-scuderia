function out = gpmlReference(doc)
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
