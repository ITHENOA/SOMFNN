clear
clc
%%
% load mgdata.dat
% time = mgdata(:,1);
% x = mgdata(:, 2);
% for t = 118:1117
%     Data(t-117,:) = [x(t-18) x(t-12) x(t-6) x(t) x(t+6)];
% end
% Xtr = Data(1:800, 1:end-1);
% Ytr = Data(1:800, end);
% Xte = Data(801:end, 1:end-1);
% Yte = Data(801:end, end);
% clear Data x t mgdata time
%%
load mackey_0.1.mat
for k = 201:3200
    Xtr(k-200,:) = [X(k-18) X(k-12) X(k-6)];
    Ytr(k-200,:) = X(k);
end
for k = 5001:5500
    Xte(k-5000,:) = [X(k-18) X(k-12) X(k-6)];
    Yte(k-5000,:) = X(k);
end
clear X k
%%
data = table2array(readtable("C:\Users\ITHENOA_PC\Desktop\MSOFNN_new\dataset\auto_mpg_data (5).xlsx"));
data(logical(sum(isnan(data),2)),:) = [];
idx=randperm(size(data,1));
Xtr = data(idx(1:196),1:7);
Xte = data(idx(197:end),1:7);
Ytr = data(idx(1:196),8);
Yte = data(idx(197:end),8);

%% construct neuro-fuzzy network
clc
net = MSOFNNplus(Xtr,Ytr,3,...
    "ActivationFunction", "sig",...
    "DensityThreshold", exp(-5),...
    "MaxEpoch", 100,...
    "BatchNormType", "none",...
    "LearningRate", 1,...
    "SolverName", "minibatchgd",...
    "WeightInitializationType", "none",...
    "DataNormalize" , "XY",...
    "MiniBatchSize", 32,...
    "adampar_beta1", 0.6,...
    "adampar_beta2", 0.8,...
    "adampar_epsilon", 1e-8,...
    "adampar_m0", 0,...
    "adampar_v0", 0,...
    "plot", 0,...
    "verbose", 1)

% Train
tic
[trained_net,best] = net.train("validationPercent",0.2);
toc

%% Test
Yte = normalize(Yte,1,"range");
[yhat,err] = trained_net.test(Xte,Yte);
disp(err)
plot(Yte)
hold on
plot(yhat')
