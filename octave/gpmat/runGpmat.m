% Run every golden through the GPmat adapter and write raw results as JSON, in the same
% shape as octave/gpml/runGpml.m; scripts/compare.py applies tolerances.
%   GPMAT=<GPmat checkout> GPSCUDERIA_ROOT=<repo> GPSCUDERIA_OUT=<file> octave-cli octave/gpmat/runGpmat.m
warning('off', 'all');
here = fileparts(mfilename('fullpath'));
root = getenv('GPSCUDERIA_ROOT'); if isempty(root), root = fileparts(fileparts(here)); end
gpmat = getenv('GPMAT'); if isempty(gpmat), error('Set GPMAT to a GPmat checkout.'); end
addpath(genpath(gpmat)); addpath([root '/octave/shims']); addpath([root '/octave/gpscuderia']);
files = dir(fullfile(root, 'examples', '*.json'));
res = struct();
for i = 1:numel(files)
  [~, name] = fileparts(files(i).name);
  try
    doc = gpscuderiaRead(fullfile(files(i).folder, files(i).name));
    evalc('m = gpscuderiaToModel(doc);');
    g = gpscuderiaReference(m, doc.x_reference.Xnew);
    res.(name) = struct('log_likelihood', g.logLikelihood, 'mean', g.mean(:)', 'var', g.var(:)');
  catch err
    if ~isempty(strfind(err.identifier, 'gpscuderia:unsupported'))
      res.(name) = struct('refused', err.message);
    else
      res.(name) = struct('error', [err.identifier ': ' err.message]);
    end
  end
end
out = struct('octave', version(), 'gpmat', getenv('GPMAT_VERSION'), 'results', res);
fid = fopen(getenv('GPSCUDERIA_OUT'), 'w'); fputs(fid, jsonencode(out)); fclose(fid);
