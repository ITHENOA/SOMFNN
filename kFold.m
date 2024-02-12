function [dataTR, dataTE, test_data] = kFold(X,Y,k,test_ratio)
arguments
    X
    Y
    k = 10
    test_ratio = 0
end
test_data = [];
[X,shuffled_idx] = shuffle(X);
Y = Y(shuffled_idx,:);
nTest = round(length(X)*test_ratio);
test_data.x = X(1:nTest,:);
test_data.y = Y(1:nTest,:);
X(1:nTest,:) = [];
Y(1:nTest,:) = [];
splitSize = round(length(X)/k);
idx = 1:length(X);
[dataTR,dataTE] = deal(cell(1,k));
for fold = 1:k
    if fold == k
        idx_test = idx((fold-1)*splitSize+1:end);
    else
        idx_test = idx((fold-1)*splitSize+1:(fold)*splitSize);
    end
    idx_train = setdiff(idx,idx_test);
    dataTR{fold}.x = X(idx_train,:);
    dataTR{fold}.y = Y(idx_train,:);
    dataTE{fold}.x = X(idx_test,:);
    dataTE{fold}.y = Y(idx_test,:);
end