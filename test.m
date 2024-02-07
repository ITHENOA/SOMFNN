clear
clc
%% Pen (10 class)
clear;clc;close all
addpath("C:\Users\ITHENOA_PC\Desktop\MSOFNN\dataset")
data = readmatrix("pen_class10.xlsx");
X = data(:,1:end-1);
Y = data(:,end);
idx = randperm(numel(Y));
Xtr = X(idx(1:7494),:);
Ytr = Y(idx(1:7494),1);
Xte = X(idx(7495:end),:);
Yte = Y(idx(7495:end),1);
clear X Y idx data
datasetName = "Pen";
%% GesPhase (5 class)
clear;clc;close all
addpath("C:\Users\ITHENOA_PC\Desktop\MSOFNN\dataset")
load("GesturePhase_class5.mat")
X = GestrurePhase(:,1:end-1);
Y = GestrurePhase(:,end);
idx = randperm(numel(Y));
Xtr = X(idx(1:6500),:);
Ytr = Y(idx(1:6500),1);
Xte = X(idx(6501:end),:);
Yte = Y(idx(6501:end),1);
clear X Y idx GestrurePhase
datasetName = "GesPhase";
%% SP500 (1 reg)
clear;clc;close all
addpath("C:\Users\ITHENOA_PC\Desktop\MSOFNN\dataset")
data = readmatrix("SP500_reg.xlsx");
data = data(:,5);
data = [data ; flipud(data)];
data = normalize(data,1,"range");
for t = 5:numel(data)-1
    X(t,:) = [data(t-4) data(t-3) data(t-2) data(t-1) data(t)];
    Y(t,1) = data(t+1);
end
Xtr = X(1:15038,:);
Ytr = Y(1:15038,1);
Xte = X(15039:end,:);
Yte = Y(15039:end,1);
clear data X Y
datasetName = "SP500";
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
datasetName = "liver";
%% KMG (3 class)
clear;clc;close all
addpath("C:\Users\ITHENOA_PC\Desktop\MSOFNN\dataset\KMG")
% addpath("C:\Users\user\Desktop\MSOFNN\dataset\KMG")
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
datasetName = "KMG";
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
datasetName = "M.Glass";
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
datasetName = "A.MPG";

%% construct neuro-fuzzy network
clc,close all
% rng(1)
net = MSOFNNplus(Xtr,Ytr,3,...
    "n_hiddenNodes","Auto",...
    "ActivationFunction", ["sigmoid","linear"],...
    "DensityThreshold", exp(-5),...
    "MaxEpoch", 200,...
    "BatchNormType", "none",...
    "LearningRate", 0.001,...
    "SolverName", "adam",...
    "WeightInitializationType", "none",...
    "DataNormalize" , "X",...
    "MiniBatchSize", 128,...
    "adampar_beta1", 0.6,...
    "adampar_beta2", 0.8,...
    "adampar_epsilon", 1e-8,...
    "adampar_m0", 0,...
    "adampar_v0", 0,...
    "Plot", 0,...
    "Verbose", 5)

excelFileName = 'results.xlsx';
%% %%%%%%%%%%%%%%%%%
clc
n_run = 10;
%%%%%%%%%%%%
for str = ["m_b","m_p","m_mus","m_mis","m_n"]
    eval(str+"_LOSS = zeros("+n_run+",1);")
    eval(str+"_MSE = zeros("+n_run+",1);")
    eval(str+"_RMSE = zeros("+n_run+",1);")
    eval(str+"_MAE = zeros("+n_run+",1);")
    eval(str+"_NDEI = zeros("+n_run+",1);")
    eval(str+"_NDEI2 = zeros("+n_run+",1);")
    eval(str+"_ACC = zeros("+n_run+",1);")
    eval(str+"_PREC = zeros("+n_run+",1);")
    eval(str+"_RECALL = zeros("+n_run+",1);")
    eval(str+"_F1SCORE = zeros("+n_run+",1);")
    eval(str+"_KAPPA = zeros("+n_run+",1);")
    eval(str+"_MCC = zeros("+n_run+",1);")
end
[texe,layers,rules,rem_rule_p,rem_rule_mus,rem_rule_mis,rem_rule_n]= deal(zeros(n_run,1));
for i = 1:n_run
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [trained_net,valdata] = net.Train(...
        "validationSplitPercent",0.2,...
        "valPerEpochFrequency",1,...
        "ApplyRuleRemover",0);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    texe(i) = toc;
    [~,metricsLast] = Test(trained_net.last, Xte, Yte);
    [~,metricsBest] = Test(trained_net.best, Xte, Yte);
    if contains(net.ProblemType,"Regression")
        if metricsBest.MSEorACC < metricsLast.MSEorACC
            TRDnet = trained_net.best;
        else
            TRDnet = trained_net.last;
        end
    else
        if metricsBest.MSEorACC > metricsLast.MSEorACC
            TRDnet = trained_net.best;
        else
            TRDnet = trained_net.last;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    net_removedRule = RuleRemover(TRDnet,valdata.x,0.5,0.5,...
        "OptimizeParams",0,...
        "OutputData_required",[],...
        "DisplayIterations","iter",...
        "PSO_MaxIterations",20,...
        "PSO_MaxStallIterations",7);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    layers(i) = TRDnet.n_Layer;
    rules(i) = mean(TRDnet.n_rulePerLayer);
    rem_rule_p(i) =  mean(net_removedRule.Percentage.n_rulePerLayer);
    rem_rule_mus(i) =  mean(net_removedRule.MeanMultiStd.n_rulePerLayer);
    rem_rule_mis(i) =  mean(net_removedRule.MeanMinesStd.n_rulePerLayer);
    rem_rule_n(i) =  mean(net_removedRule.new.n_rulePerLayer);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,m_b] = Test(TRDnet, Xte, Yte, "Plot",0,"MetricReport","all");
    [~,m_p] = Test(net_removedRule.Percentage, Xte, Yte, "Plot",0,"MetricReport","all");
    [~,m_mus] = Test(net_removedRule.MeanMultiStd, Xte, Yte, "Plot",0,"MetricReport","all");
    [~,m_mis] = Test(net_removedRule.MeanMinesStd, Xte, Yte, "Plot",0,"MetricReport","all");
    [~,m_n] = Test(net_removedRule.new, Xte, Yte, "Plot",0,"MetricReport","all");
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for str = ["m_b","m_p","m_mus","m_mis","m_n"]
        if contains(net.ProblemType,"Regression")
            eval(str+"_LOSS(i) = "+str+".LOSS;")
            eval(str+"_MSE(i) = "+str+".MSE;")
            eval(str+"_RMSE(i) = "+str+".RMSE;")
            eval(str+"_MAE(i) = "+str+".MAE;")
            eval(str+"_NDEI(i) = "+str+".NDEI;")
            eval(str+"_NDEI2(i) = "+str+".NDEI2;")
        else
            eval(str+"_ACC(i) = "+str+".ACC;")
            eval(str+"_PREC(i) = "+str+".PREC;")
            eval(str+"_RECALL(i) = "+str+".RECALL;")
            eval(str+"_F1SCORE(i) = "+str+".F1SCORE;")
            eval(str+"_KAPPA(i) = "+str+".KAPPA;")
            eval(str+"_MCC(i) = "+str+".MCC;")
        end
    end
end
% save to exel
run = (1:n_run)';
if contains(net.ProblemType,"Regression")
    tbl = table(run,texe,layers, ...
        rules,m_b_LOSS,m_b_MSE,m_b_RMSE,m_b_MAE,m_b_NDEI,m_b_NDEI2, ...
        rem_rule_p,m_p_LOSS,m_p_MSE,m_p_RMSE,m_p_MAE,m_p_NDEI,m_p_NDEI2, ...
        rem_rule_mus,m_mus_LOSS,m_mus_MSE,m_mus_RMSE,m_mus_MAE,m_mus_NDEI,m_mus_NDEI2, ...
        rem_rule_mis,m_mis_LOSS,m_mis_MSE,m_mis_RMSE,m_mis_MAE,m_mis_NDEI,m_mis_NDEI2, ...
        rem_rule_n,m_n_LOSS,m_n_MSE,m_n_RMSE,m_n_MAE,m_n_NDEI,m_n_NDEI2);
else
    tbl = table(run,texe,layers, ...
        rules,m_b_ACC,m_b_PREC,m_b_RECALL,m_b_F1SCORE,m_b_KAPPA,m_b_MCC, ...
        rem_rule_p,m_p_ACC,m_p_PREC,m_p_RECALL,m_p_F1SCORE,m_p_KAPPA,m_p_MCC, ...
        rem_rule_mus,m_mus_ACC,m_mus_PREC,m_mus_RECALL,m_mus_F1SCORE,m_mus_KAPPA,m_mus_MCC, ...
        rem_rule_mis,m_mis_ACC,m_mis_PREC,m_mis_RECALL,m_mis_F1SCORE,m_mis_KAPPA,m_mis_MCC, ...
        rem_rule_n,m_n_ACC,m_n_PREC,m_n_RECALL,m_n_F1SCORE,m_n_KAPPA,m_n_MCC);
end
writetable(tbl, excelFileName, 'Sheet', datasetName);

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
