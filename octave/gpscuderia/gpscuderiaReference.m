function ref = gpscuderiaReference(model, Xnew)

% GPSCUDERIAREFERENCE Log likelihood and predictions for conformance tests.
% FORMAT
% DESC returns struct with logLikelihood, mean and var (noisy predictive
% variance, matching GPy predict) at Xnew.
% ARG model : GPmat model.
% ARG Xnew : test inputs.
% RETURN ref : structure of results.
%
% COPYRIGHT : Neil D. Lawrence, 2026

[mu, varsigma] = gpPosteriorMeanVar(model, Xnew);
ll = modelLogLikelihood(model);
% GPmat's sparse bound can come back complex: logdet() takes log of a Cholesky-
% like factor whose diagonal has negative entries, adding i*pi per sign flip.
% Only sign flips are benign, so require the imaginary part to be k*pi.
if abs(imag(ll)/pi - round(imag(ll)/pi)) > 1e-9
  error('gpscuderia:complexLL', 'log likelihood has imaginary part %g (not k*pi)', imag(ll));
end
% GPmat's objective has no -N log(scale) Jacobian (learnScales aside); the wire
% reports log p(y), so add it here.
ll = ll - model.N*sum(log(model.scale));
ref.logLikelihood = real(ll);
ref.mean = mu(:)';
ref.var = varsigma(:)';
