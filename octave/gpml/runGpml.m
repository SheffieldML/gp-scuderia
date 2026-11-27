% Run every golden through GPML and write a JSON file of raw results (no judgement here:
% scripts/compare.py applies tolerances). Usage under octave-cli with GPML and GPSCUDERIA_OUT set:
%   GPML=<gpml dir> GPSCUDERIA_ROOT=<repo> GPSCUDERIA_TASK=<task.json> GPSCUDERIA_OUT=<file> octave-cli octave/gpml/runGpml.m
warning('off', 'all');
here = fileparts(mfilename('fullpath'));
root = getenv('GPSCUDERIA_ROOT'); if isempty(root), root = fileparts(fileparts(here)); end
gp_dir = getenv('GPML'); if isempty(gp_dir), error('Set GPML to a GPML checkout.'); end
evalc('run(fullfile(gp_dir, ''startup.m''))');
addpath(here);
task = jsondecode(fileread(getenv('GPSCUDERIA_TASK')));
files = task.tasks;
res = struct();
for i = 1:numel(files)
  name = files(i).name; golden = files(i).golden;
  doc = jsondecode(fileread(golden));
  if isfield(doc, 'x_reference') == 0 && isfield(doc, 'x0x2Dreference'), doc.x_reference = doc.x0x2Dreference; end
  doc.data.X = reshape(doc.data.X, size(doc.data.X, 1), []);
  try
    if isfield(doc, 'inducing_inputs'), doc.inducing_inputs = reshape(doc.inducing_inputs, size(doc.inducing_inputs, 1), []); end
    evalc('r = gpmlReference(doc);');
    res.(name) = r;
  catch err
    if strcmp(err.identifier, 'gpscuderia:unsupported')
      res.(name) = struct('refused', err.message);
    else
      res.(name) = struct('error', [err.identifier ': ' err.message]);
    end
  end
end
out = struct('schema', 'gp-scuderia-result/1', 'octave', version(), 'gpml', getenv('GPML_VERSION'), 'results', res);
fid = fopen(getenv('GPSCUDERIA_OUT'), 'w'); fputs(fid, jsonencode(out)); fclose(fid);
