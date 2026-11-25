function doc = gpscuderiaRead(fileName)

% GPSCUDERIARead Read an interchange document from its JSON twin.
% FORMAT
% DESC reads a gp-scuderia model document (JSON; Octave and MATLAB have no
% YAML reader) and returns the decoded structure. Matrices arrive as
% numeric arrays; kernel part lists are normalised to a cell array of structs.
% ARG fileName : path to the .json file.
% RETURN doc : decoded document.
%
% SEEALSO : gpscuderiaToModel, gpscuderiaFromModel
%
% COPYRIGHT : Neil D. Lawrence, 2026

fid = fopen(fileName, 'r');
if fid < 0
  error('gpscuderia:io', 'cannot open %s', fileName);
end
txt = fread(fid, Inf, 'char=>char')';
fclose(fid);
doc = jsondecode(txt);
doc.kernel = normaliseKernel(doc.kernel);


function k = normaliseKernel(k)
if isfield(k, 'parts')
  p = k.parts;
  if isstruct(p)
    q = cell(numel(p), 1);
    for i = 1:numel(p), q{i} = p(i); end
    p = q;
  end
  for i = 1:numel(p), p{i} = normaliseKernel(p{i}); end
  k.parts = p;
end
