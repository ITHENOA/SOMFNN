clear
clc
%% liver (Binery classification)
clear;clc;close all
data = table2array(readtable("C:\Users\ITHENOA_PC\Desktop\MSOFNN\dataset\liver_disorders.xlsx"));
Y = data(:,end);
X = data(:,1:end-1);
idx = randperm(size(data,1));
idxTr = idx(1:200);
idxTe = idx(201:end);
Xtr = X(idxTr,:);
Ytr = Y(idxTr);
Xte = X(idxTe,:);
Yte = Y(idxTe);
clear data X Y idx idxTr idxTe
%% KMG (3 class)
clear;clc;close all
addpath("C:\Users\ITHENOA_PC\Desktop\MSOFNN\dataset\KMG")
load fieldAmp.mat
load fieldAmp_t.mat
% Xte = fieldAmp_t(:,1:end-1);
% Yte = fieldAmp_t(:,end);
idx = randperm(size(fieldAmp,1));
Xtr = fieldAmp(idx(1:10000),1:end-1);
Ytr = fieldAmp(idx(1:10000),end);
Xte = fieldAmp(idx(10001:end),1:end-1);
Yte = fieldAmp(idx(10001:end),end);
clear fieldAmp fieldAmp_t idx
%% Mackey Glass (1 reg)
clear;clc;close all
load MackeyGlassNew.mat
data(:,1) = [];
for t = 201:3200
    Xtr(t-200,:) = [data(t-18) data(t-12) data(t-6) data(t)];
    Ytr(t-200,:) = data(t + 85);
end
for t = 5001:5500
    Xte(t-5000,:) = [data(t-18) data(t-12) data(t-6) data(t)];
    Yte(t-5000,:) = data(t + 85);
end
clear data t
%% Mackey Glass (easy)
clear;clc;close all
load mackey_0.1.mat
for k = 201:3200
    Xtr(k-200,:) = [X(k-24) X(k-18) X(k-12) X(k-6)];
    Ytr(k-200,:) = X(k);
end
for k = 5001:5500
    Xte(k-5000,:) = [X(k-24) X(k-18) X(k-12) X(k-6)];
    Yte(k-5000,:) = X(k);
end
clear X k
%% Auto MPG (1 reg)
clear;clc;close all
data = table2array(readtable("C:\Users\ITHENOA_PC\Desktop\MSOFNN\dataset\auto_mpg_data (5).xlsx"));
% data = table2array(readtable("C:\Users\hojja\Desktop\MSOFNN\dataset\auto_mpg_data (5).xlsx"));
% data = table2array(readtable("C:\Users\Cib-Sabz\Documents\MATLAB\Fuzzy Logic - Fall 2023\Final Project\dataset\auto_mpg_data (5).xlsx"));
% data = table2array(readtable("C:\Users\Cib-Sabz\Documents\MATLAB\Fuzzy Logic - Fall 2023\Final Project\MSOFNN-main2\dataset\auto_mpg_data (5).xlsx"));
data(logical(sum(isnan(data),2)),:) = [];
idx=randperm(size(data,1));
Xtr = data(idx(1:196),1:7);
Xte = data(idx(197:end),1:7);
Ytr = data(idx(1:196),8);
Yte = data(idx(197:end),8);
clear data idx
%% SP500
clear;clc;close all
data = table2array(readtable("C:\Users\hojja\Desktop\MSOFNN\dataset\SP500.csv"))
%% construct neuro-fuzzy network
clc,close all
% rng(1)
for i = 1:1
    i
net = MSOFNNplus(Xtr,Ytr,2,...
    "n_hiddenNodes","Auto",...
    "ActivationFunction", ["sigmoid"],...
    "DensityThreshold", exp(-5),...
    "MaxEpoch", 500,...
    "BatchNormType", "none",...
    "LearningRate", 0.01,...
    "SolverName", "adam",...
    "WeightInitializationType", "none",...
    "DataNormalize" , "XY",...
    "MiniBatchSize", 128,...
    "adampar_beta1", 0.6,...
    "adampar_beta2", 0.8,...
    "adampar_epsilon", 1e-8,...
    "adampar_m0", 0,...
    "adampar_v0", 0,...
    "Plot", 1,...
    "Verbose", 5)

% Train
% tic
trained_net = net.Train(...
    "validationSplitPercent",0.2,...
    "valPerEpochFrequency",5,...
    "ApplyRuleRemover",0);
% toc

%

[~,metrics] = Test(trained_net.best, Xte, Yte, "Plot",1,"MetricReport","all");
disp(metrics)
% LOSS(i) = err.LOSS;
% MSE(i) = err.MSE;
% RMSE(i) = err.RMSE;
% NDEI(i) = err.NDEI;
% NDEI2(i) = err.NDEI2;
end
% loss=mean(LOSS)
% mse=mean(MSE)
% rmse=mean(RMSE)
% ndei=mean(NDEI)
% ndei2=mean(NDEI2)
% [~,err] = Test(trained_net.last, Xte, Yte, "Plot",1);disp(err)
% [~,err] = Test(trained_net.best, Xte, Yte, "Plot",1);disp(err)

idx = randperm(size(Xtr,1));
n_val = round(size(Xtr,1) * 0.2);
Xval = Xtr(idx(1:n_val),:);
Yval = Ytr(idx(1:n_val));
%%
net_removedRule = RuleRemover(trained_net.best,Xval,0.5,0.5,...
    "OptimizeParams",0,...
    "OutputData_required",Yval,...
    "DisplayIterations","iter",...
    "PSO_MaxIterations",20,...
    "PSO_MaxStallIterations",7)


% Test
figure
plot(Yte,DisplayName='Ref')
hold on
networks = [trained_net.last, ...
        trained_net.best, ...
        net_removedRule.Percentage, ...
        net_removedRule.MeanMultiStd, ...
        net_removedRule.MeanMinesStd, ...
        net_removedRule.new];
name = ["last","best","perc","mean*std","mean-std","new"];
markers = ["*","o","+","square",">","x","."];
for i = 1:numel(networks)
    [yhat,err] = Test(networks(i), Xte, Yte, "Plot",0,"MetricReport","all");
    disp(err)
    disp(networks(i).n_rulePerLayer)
    plot(yhat,DisplayName=name(i),Marker=markers(i),MarkerIndices=1:10:numel(Yte),LineWidth=1.5)
end
legend('show')

% [yhat_best,err] = Test(trained_net.best, Xte, Yte, "Plot",0);
% disp(err)
% disp(trained_net.last.n_rulePerLayer)
% [yhat_1,err] = Test(net_removedRule.Percentage, Xte, Yte, "Plot",0);
% disp(err)
% disp(trained_net.last.n_rulePerLayer)
% [yhat_2,err] = Test(net_removedRule.MeanMultiStd, Xte, Yte, "Plot",0);
% disp(err)
% disp(trained_net.last.n_rulePerLayer)
% [yhat_3,err] = Test(net_removedRule.MeanMinesStd, Xte, Yte, "Plot",0);
% disp(err)
% disp(trained_net.last.n_rulePerLayer)
% % [yhat_last,err] = trained_net.last.Test(Xte,Yte,"Plot",0);
% % disp(err)
% % [yhat_best,err] = trained_net.best.Test(Xte,Yte);
% % disp(err)
% % figure
% 
% plot(yhat_last',DisplayName='last')
% plot(yhat_best',DisplayName='best')
% plot(yhat_1',DisplayName='ma')
% plot(yhat_2',DisplayName='per')
% plot(yhat_3',DisplayName='alaki')
% legend('show')
% % legend("Yte","yhat-last","yhat-best")
