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
function v = ifelse_(c, a, b), if c, v = a; else v = b; end, end

names = {'ftc_rbf_white', 'dtcvar_ard_data'};
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
  l0 = doc.kernel.parts{1}; l1 = back.kernel.parts{1};
  bad += check(all(abs(l0.lengthscale(:) - l1.lengthscale(:)) < 1e-12) && abs(l0.variance - l1.variance) < 1e-12, ...
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
d = base; d.kernel.type = 'kern.prod';
bad += check(rejects(d), 'rejects kern.prod (not yet supported)');
exit(bad > 0);
