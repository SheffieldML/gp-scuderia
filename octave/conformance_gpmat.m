% Conformance of the GPmat adapter against the goldens' x-reference (GPy oracle).
% Run via scripts/conformance.sh (sets GPMAT and the dyld path), under octave-cli.
warning('off', 'all');
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
gpmat = getenv('GPMAT'); if isempty(gpmat), gpmat = [getenv('HOME') '/SheffieldML/GPmat']; end
addpath(genpath(gpmat)); addpath([here '/shims']); addpath([here '/gpscuderia']);
bad = 0;
function r = check(cond, msg)
  printf('%s%s\n', ifelse_(cond, 'ok   ', 'FAIL '), msg); r = ~cond;
end
function L = leafList(node)
if isfield(node, 'parts')
  L = {};
  for k = 1:numel(node.parts), L = [L, leafList(node.parts{k})]; end
else
  L = {node};
end
end
function v = ifelse_(c, a, b), if c, v = a; else v = b; end, end

names = {'ftc_rbf_white', 'dtcvar_ard_data', 'ftc_prod_matern52', 'fitc_matern52'};
for i = 1:numel(names)
  n = names{i};
  doc = gpscuderiaRead([root '/examples/' n '.json']);
  ref = doc.x_reference; tol = ref.tolerance;
  evalc('m = gpscuderiaToModel(doc);');
  g = gpscuderiaReference(m, ref.Xnew);
  dLL = abs(g.logLikelihood - ref.log_likelihood);
  dm = max(abs(g.mean(:) - ref.mean(:))); dv = max(abs(g.var(:) - ref.var(:)));
  bad += check(dLL <= tol.log_likelihood && dm <= tol.mean && dv <= tol.var, ...
    sprintf('%s matches GPy: dLL=%.2g dmean=%.2g dvar=%.2g', n, dLL, dm, dv));
  back = gpscuderiaFromModel(m, doc);
  L0 = leafList(doc.kernel); L1 = leafList(back.kernel);
  same = numel(L0) == numel(L1);
  for k = 1:min(numel(L0), numel(L1))
    same = same && strcmp(L0{k}.type, L1{k}.type) && abs(L0{k}.variance - L1{k}.variance) < 1e-12;
    if isfield(L0{k}, 'lengthscale')
      same = same && all(abs(L0{k}.lengthscale(:) - L1{k}.lengthscale(:)) < 1e-12);
    end
  end
  bad += check(same, ...
    sprintf('%s kernel round-trips through GPmat storage', n));
  bad += check(abs(back.likelihood.variance - doc.likelihood.variance) < 1e-12, ...
    sprintf('%s likelihood variance round-trips', n));
end

% Negative tests.
base = gpscuderiaRead([root '/examples/dtcvar_ard_data.json']);
function ok = rejects(d)
  ok = false;
  try, evalc('gpscuderiaToModel(d);'); catch, ok = true; end
end
d = base; d.kernel.parts{1}.type = 'kern.nonesuch';
bad += check(rejects(d), 'rejects unknown kernel type');
d = base; d.data.X = d.data.X(:, 1);
bad += check(rejects(d), 'rejects data width != input_dim');
d = base; d.schema_version = '9.9';
bad += check(rejects(d), 'rejects unsupported schema_version');
d = base; d = rmfield(d, 'data');
bad += check(rejects(d), 'rejects missing data');
d = base; d.kernel.type = 'kern.mystery';
bad += check(rejects(d), 'rejects unknown kernel type');
exit(bad > 0);
