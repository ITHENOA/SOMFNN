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
    Xtr(k-200,:) = [X(k-18) X(k-12) X(k-6) X(k)];
    Ytr(k-200,:) = X(k-85);
end
for k = 5001:5500
    Xte(k-5000,:) = [X(k-18) X(k-12) X(k-6) X(k)];
    Yte(k-5000,:) = X(k-85);
end
clear X k
%%
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
%%
data = table2array(readtable("C:\Users\ITHENOA_PC\Desktop\MSOFNN_new\dataset\auto_mpg_data (5).xlsx"));
% data = table2array(readtable("C:\Users\Cib-Sabz\Documents\MATLAB\Fuzzy Logic - Fall 2023\Final Project\dataset\auto_mpg_data (5).xlsx"));
% data = table2array(readtable("C:\Users\Cib-Sabz\Documents\MATLAB\Fuzzy Logic - Fall 2023\Final Project\MSOFNN-main2\dataset\auto_mpg_data (5).xlsx"));
data(logical(sum(isnan(data),2)),:) = [];
idx=randperm(size(data,1));
Xtr = data(idx(1:196),1:7);
Xte = data(idx(197:end),1:7);
Ytr = data(idx(1:196),8);
Yte = data(idx(197:end),8);

%% construct neuro-fuzzy network
clc,close all
% rng(1)
net = MSOFNNplus(Xtr,Ytr,3,...
    "ActivationFunction", ["sig"],...
    "DensityThreshold", exp(-5),...
    "MaxEpoch", 30,...
    "BatchNormType", "none",...
    "LearningRate", 1,...
    "SolverName", "minibatch",...
    "WeightInitializationType", "none",...
    "DataNormalize" , "X",...
    "MiniBatchSize", 64,...
    "adampar_beta1", 0.6,...
    "adampar_beta2", 0.8,...
    "adampar_epsilon", 1e-8,...
    "adampar_m0", 0,...
    "adampar_v0", 0,...
    "Plot", 0,...
    "Verbose", 0)

% Train
tic
trained_net = net.Train(...
    "validationPercent",0.2,...
    "valPerEpochFrequency",5,...
    "ApplyRuleRemover",0)
toc

[~,err] = Test(trained_net.last, Xte, Yte, "Plot",1);disp(err)
[~,err] = Test(trained_net.best, Xte, Yte, "Plot",1);disp(err)

idx = randperm(size(Xtr,1));
n_val = round(size(Xtr,1) * 0.2);
Xval = Xtr(idx(1:n_val),:);
Yval = Ytr(idx(1:n_val));
%%
net_removedRule = RuleRemover(trained_net.best,Xval,0.5,0.5,0.5,...
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
    [yhat,err] = Test(networks(i), Xte, Yte, "Plot",0);
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
