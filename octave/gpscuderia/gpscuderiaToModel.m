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
leaves = kernLeaves(doc.kernel);
names = cell(1, numel(leaves));
for i = 1:numel(leaves)
  names{i} = gpmatName(leaves{i}, d);
end
options = gpOptions(approx);
if numel(names) == 1, options.kern = names{1}; else options.kern = names; end
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

% Kernel parameters (component structures).
for i = 1:numel(leaves)
  model.kern = setLeaf(model.kern, i, numel(leaves), leaves{i});
end
model.kern = kernExpandParam(model.kern, kernExtractParam(model.kern));

% Zero mean, unit scale: wire semantics.
model.bias = zeros(1, size(Y, 2));
model.scale = ones(1, size(Y, 2));
model.m = gpComputeM(model);
model = gpExpandParam(model, gpExtractParam(model));


function leaves = kernLeaves(node)
if isfield(node, 'parts')
  if ~strcmp(node.type, 'kern.add')
    error('gpscuderia:unsupported', '%s not supported by GPmat adapter', node.type);
  end
  leaves = {};
  for i = 1:numel(node.parts)
    leaves = [leaves, kernLeaves(node.parts{i})]; %#ok<AGROW>
  end
else
  leaves = {node};
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
  case 'kern.matern32', n = 'matern32';
  otherwise
    error('gpscuderia:unsupported', 'GPmat adapter: unsupported %s', leaf.type);
end


function c = setLeaf(kern, i, n, leaf)
c = kern;
if n == 1, comp = kern; else comp = kern.comp{i}; end
comp.variance = leaf.variance;
if isfield(comp, 'inputScales')
  comp.inputScales = (1 ./ leaf.lengthscale(:)').^2;
elseif isfield(comp, 'inverseWidth')
  comp.inverseWidth = 1/leaf.lengthscale^2;
elseif strcmp(leaf.type, 'kern.matern32')
  comp.lengthScale = leaf.lengthscale;
end
if n == 1, c = comp; else c.comp{i} = comp; end
