function model = gpscuderiaToModel(doc)

% GPSCUDERIATOMODEL Build a GPmat GP model from an interchange document.
% FORMAT
% DESC converts wire parameters (natural, positive) to GPmat storage:
% rbf inverseWidth = 1/lengthscale^2, rbfard2 inputScales = 1/lengthscale.^2,
% beta = 1/likelihood variance. The wire format is zero-mean, so GPmat's
% default bias = mean(y) is overridden with zero.
% ARG doc : document from gpscuderiaRead.
% RETURN model : GPmat model with approx and parameters set.
%
% SEEALSO : gpscuderiaRead, gpscuderiaFromModel
%
% COPYRIGHT : Neil D. Lawrence, 2026

if ~strcmp(doc.schema_version, '0.1')
  error('gpscuderia:version', 'unsupported schema_version %s', doc.schema_version);
end
if ~isfield(doc, 'data')
  error('gpscuderia:data', 'GPmat adapter needs embedded data');
end
d = doc.input_dim;
X = doc.data.X; Y = doc.data.Y;
if size(X, 2) ~= d
  error('gpscuderia:dims', 'data.X width %d != input_dim %d', size(X, 2), d);
end
approx = 'ftc';
if isfield(doc, 'approximation'), approx = doc.approximation; end
options = gpOptions(approx);
options.kern = buildKern(doc.kernel, X, d);
if ~strcmp(doc.likelihood.type, 'likelihoods.gaussian')
  error('gpscuderia:unsupported', ['GPmat adapter: %s is not supported (stacked multi-output ' ...
        'models with per-output noise are GPy and GPc only)'], doc.likelihood.type);
end
sigma2 = doc.likelihood.variance;
options.beta = 1/sigma2;
if strcmp(approx, 'ftc')
  options.optimiseBeta = true;   % so gpCreate stores beta; frozen below
else
  Z = doc.inducing_inputs;
  options.numActive = size(Z, 1);
  options.fixInducing = true;
  options.fixIndices = 1:size(Z, 1);
  options.optimiseBeta = false;
end
model = gpCreate(d, size(Y, 2), X, Y, options);
if strcmp(approx, 'ftc'), model.optimiseBeta = false; else model.X_u = Z; end

% Target transform: no field means zero mean and unit scale. GPmat stores
% m = (y - bias)./scale, which is what mean.constant and normalizer describe.
if isfield(doc, 'mean') && isfield(doc, 'normalizer')
  error('gpscuderia:unsupported', 'mean and normalizer together are not in schema 0.1');
end
bias = 0; scale = 1;
if isfield(doc, 'mean')
  if ~strcmp(doc.mean.type, 'mean.constant')
    error('gpscuderia:unsupported', '%s not supported by GPmat adapter', doc.mean.type);
  end
  bias = doc.mean.value;
end
if isfield(doc, 'normalizer')
  switch doc.normalizer.type
    case 'normalizer.center',      bias = doc.normalizer.mean;
    case 'normalizer.standardize', bias = doc.normalizer.mean; scale = doc.normalizer.std;
    otherwise
      error('gpscuderia:unsupported', '%s not supported by GPmat adapter', doc.normalizer.type);
  end
end
model.bias = bias*ones(1, size(Y, 2));
model.scale = scale*ones(1, size(Y, 2));
model.m = gpComputeM(model);
model = gpExpandParam(model, gpExtractParam(model));


function [kern, params] = buildKern(node, X, d)
% Recursive kernel construction: leaves carry their parameters, composites
% (kern.add -> cmpnd, kern.prod -> tensor) re-initialise their children, so the
% concatenated parameter vector is expanded after assembly.
if isfield(node, 'parts')
  switch node.type
    case 'kern.add',  ktype = 'cmpnd';  initFn = @cmpndKernParamInit;
    case 'kern.prod', ktype = 'tensor'; initFn = @tensorKernParamInit;
    otherwise
      error('gpscuderia:unsupported', '%s not supported by GPmat adapter', node.type);
  end
  kern = kernCreate(X, {ktype, 'rbf'});
  kern.comp = cell(1, numel(node.parts));
  params = [];
  for i = 1:numel(node.parts)
    [c, p] = buildKern(node.parts{i}, X, d);
    c.index = [];
    kern.comp{i} = c;
    params = [params, p]; %#ok<AGROW>
  end
  kern = initFn(kern);
  kern = kernExpandParam(kern, params);
  params = kernExtractParam(kern);
else
  if isfield(node, 'active_dims')
    error('gpscuderia:unsupported', ['GPmat adapter: active_dims on %s is not supported ' ...
          '(coregionalised ICM/LCM models are GPy and GPc only)'], node.type);
  end
  kern = kernCreate(X, gpmatName(node, d));
  kern = setLeaf(kern, node);
  params = kernExtractParam(kern);
end


function n = gpmatName(leaf, d)
switch leaf.type
  case 'kern.rbf'
    if numel(leaf.lengthscale) > 1 || (isfield(leaf, 'ARD') && leaf.ARD && d > 1)
      n = 'rbfard2';
    else
      n = 'rbf';
    end
  case 'kern.white', n = 'white';
  case 'kern.bias',  n = 'bias';
  case 'kern.linear', n = 'lin';
  case {'kern.matern32', 'kern.matern52'}
    if numel(leaf.lengthscale) > 1
      error('gpscuderia:unsupported', 'GPmat has no ARD %s kernel', leaf.type);
    end
    n = leaf.type(6:end);
  otherwise
    error('gpscuderia:unsupported', 'GPmat adapter: unsupported %s', leaf.type);
end


function comp = setLeaf(comp, leaf)
comp.variance = leaf.variance;
if isfield(comp, 'inputScales')
  comp.inputScales = (1 ./ leaf.lengthscale(:)').^2;
elseif isfield(comp, 'inverseWidth')
  comp.inverseWidth = 1/leaf.lengthscale^2;
elseif any(strcmp(leaf.type, {'kern.matern32', 'kern.matern52'}))
  comp.lengthScale = leaf.lengthscale;
end
