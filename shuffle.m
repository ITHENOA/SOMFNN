function [x,shuffled_idx] = shuffle(x,dim)
if nargin < 2, dim = 1; end
maxDim = numel(size(x));
idx = cell(1,maxDim);
[idx{:}] = deal(':');
shuffled_idx = randperm(size(x,dim));
idx{dim} = shuffled_idx;
x = x(idx{:});