function g = gpscuderiaGradient(model, doc)

% GPSCUDERIAGRADIENT dLL/d(wire parameter) for a GPmat model from gpscuderiaToModel.
% FORMAT
% DESC GPmat differentiates the objective with respect to its transformed
% parameters (log of each positive parameter, see kernExtractParam), so the wire
% derivative needs the chain rule through GPmat's own storage:
%   rbf        [inverseWidth variance],   inverseWidth = 1/lengthscale^2
%   rbfard2    [variance inputScales...], inputScales = 1/lengthscale.^2
%   matern     [lengthScale variance]
%   white/bias [variance]
%   noise      beta = 1/variance
% Only the exp transform is handled; anything else raises gpscuderia:unsupported.
% ARG model : GPmat model.
% ARG doc : the interchange document it came from.
% RETURN g : containers.Map from wire path to derivative of the log likelihood.
%
% COPYRIGHT : Neil D. Lawrence, 2026

if ~strcmp(doc.likelihood.type, 'likelihoods.gaussian')
  error('gpscuderia:unsupported', 'gradient: %s is outside the gradient contract', doc.likelihood.type);
end
model.optimiseBeta = true;   % include beta in the parameter vector
gp = gpLogLikeGradients(model);
gp = real(gp(:)');
nk = model.kern.nParams;
checkExp(model.kern);
gk = gp(1:nk); gbeta = gp(nk + 1);
g = containers.Map();
pos = 0;
[g, pos] = leaf(g, model.kern, doc.kernel, 'kernel', gk, pos);
if pos ~= nk, error('gpscuderia:gradient', 'consumed %d of %d kernel parameters', pos, nk); end
sigma2 = doc.likelihood.variance;
% beta = 1/sigma2, d(log beta)/d(sigma2) = -1/sigma2
g('likelihood.variance') = -gbeta / sigma2;
end


function checkExp(kern)
if isfield(kern, 'comp') && ~isempty(kern.comp)
  for i = 1:numel(kern.comp), checkExp(kern.comp{i}); end
end
for i = 1:numel(kern.transforms)
  if ~any(strcmp(kern.transforms(i).type, {'exp', 'positive'}))
    error('gpscuderia:unsupported', 'gradient: %s transform', kern.transforms(i).type);
  end
end
end


function [g, pos] = leaf(g, kern, node, path, gk, pos)
% gk is the gradient with respect to log parameters, so d/dx = (d/dlog x)/x.
if isfield(node, 'parts')
  parts = node.parts; if isstruct(parts), parts = num2cell(parts); end
  for i = 1:numel(parts)
    [g, pos] = leaf(g, kern.comp{i}, parts{i}, sprintf('%s.parts.%d', path, i - 1), gk, pos);
  end
  return
end
switch kern.type
  case 'rbf'
    % params [inverseWidth variance]; inverseWidth = ls^-2, so d log(invW)/d ls = -2/ls
    ls = node.lengthscale;
    g([path '.lengthscale']) = -2 * gk(pos + 1) / ls;
    g([path '.variance']) = gk(pos + 2) / node.variance; pos = pos + 2;
  case 'rbfard2'
    n = numel(kern.inputScales); ls = node.lengthscale(:)';
    g([path '.variance']) = gk(pos + 1) / node.variance;
    g([path '.lengthscale']) = -2 * gk(pos + 1 + (1:n)) ./ ls; pos = pos + 1 + n;
  case {'matern32', 'matern52'}
    g([path '.lengthscale']) = gk(pos + 1) / node.lengthscale;
    g([path '.variance']) = gk(pos + 2) / node.variance; pos = pos + 2;
  case {'white', 'bias'}
    g([path '.variance']) = gk(pos + 1) / node.variance; pos = pos + 1;
  otherwise
    error('gpscuderia:unsupported', 'gradient: GPmat kernel %s', kern.type);
end
end
