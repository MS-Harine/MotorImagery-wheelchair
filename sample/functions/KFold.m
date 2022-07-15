function folds = KFold(samples, fold, varargin)

p = inputParser;
addRequired(p, 'samples', @isscalar);
addRequired(p, 'fold', @isscalar);
addParameter(p, 'shuffle', false, @islogical);
addParameter(p, 'dropLast', true, @islogical);
parse(p, samples, fold, varargin{:});

nSample = p.Results.samples;
nFold = p.Results.fold;
shuffle = p.Results.shuffle;
dropLast = p.Results.dropLast;

iterList = 1:nSample;
lengthPerFold = floor(nSample / nFold);
if shuffle
    iterList = iterList(randperm(nSample));
end

folds = cell(nFold, 1);
for i = 1:nFold
    folds{i} = iterList((i - 1) * lengthPerFold + 1 : i * lengthPerFold);
end
if ~dropLast && nSample > nFold * lengthPerFold
    folds{end + 1} = iterList(nFold * lengthPerFold + 1 : end);
end

if length(folds{end}) == length(folds{end - 1})
    folds = cell2mat(folds).';
end

end

