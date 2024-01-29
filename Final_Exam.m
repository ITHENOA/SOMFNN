clc;clear;close all;
data_disp=1;





%% ------------------------------- Part 1 ---------------------------------------

% Set random seed for reproducibility
rng(42);

% Generate 2D data points with 5 clusters
numDataPoints = 500;
numClusters = 5;

% Generate random cluster centers
clusterCenters = rand(numClusters, 2) * 10;

% Generate data points around the cluster centers
dataPoints = [];
labels = [];

for i = 1:numClusters
    clusterPoints = randn(numDataPoints/numClusters, 2) + clusterCenters(i, :);
    dataPoints = [dataPoints; clusterPoints];
    labels = [labels; repmat(i, numDataPoints/numClusters, 1)];
end

% Plot the generated data points
if data_disp==1
    scatter(dataPoints(:, 1), dataPoints(:, 2), 50, labels, 'filled');
    title('Generated 2D Data with 5 Clusters');
    xlabel('X-axis');
    ylabel('Y-axis');
    colormap(jet);
    colorbar;
end

%% ------------------------------- Date praperation ---------------------------------------
mixxed_idx=randperm(numDataPoints,numDataPoints);
input_tr=dataPoints(mixxed_idx(1:4/5*length(dataPoints)),:);
input_ts=dataPoints(mixxed_idx(4/5*length(dataPoints)+1:end),:);
output_tr=labels(mixxed_idx(1:4/5*length(dataPoints)),:);
output_ts=labels(mixxed_idx(4/5*length(dataPoints)+1:end),:);

%% ------------------------------- Part 2 ---------------------------------------
opt = genfisOptions('FCMClustering','FISType','mamdani');
opt.Verbose = 0;
fis1 = genfis(input_tr,output_tr,opt);
fis1_tuned=tunefis_k(fis1,input_tr,output_tr);
error_fis1_tuned=EP(fis1_tuned,input_ts,output_ts,0);

%% ------------------------------- Part 3 ---------------------------------------
opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.Verbose = 0;
fis2 = genfis(input_tr,output_tr,opt);
fis2_tuned=tunefis_k(fis2,input_tr,output_tr);
error_fis2_tuned=EP(fis2_tuned,input_ts,output_ts,1);

%% ------------------------------- Part 4 ---------------------------------------
fis3_mamdani = convertToType2(fis1);
fis3_sugeno = convertToType2(fis2);
fis3_mamdani_tuned=tunefis_k(fis3_mamdani,input_tr,output_tr);
fis3_sugeno_tuned=tunefis_k(fis3_sugeno,input_tr,output_tr);
error_fis3_mamdani_tuned=EP(fis3_mamdani_tuned,input_ts,output_ts,0);
error_fis3_sugeno_tuned=EP(fis3_sugeno_tuned,input_ts,output_ts,1);



