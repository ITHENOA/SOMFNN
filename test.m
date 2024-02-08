clear
clc
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excelFileName = 'results.xlsx';
datasetDirectory = "./dataset";
n_run = 10;
% for datasetName = ["sp500", "mglass", "ampg"] % regression
for datasetName = ["gesphase"] % classification
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [TR,TE] = datasetBenchmark(datasetName,datasetDirectory);

    % construct neuro-fuzzy network
    clc,close all
    % rng(1)
    net = MSOFNNplus(TR.x,TR.y,2,...
        "n_hiddenNodes",5,...
        "ActivationFunction", ["sig","lin"],...
        "DensityThreshold", exp(-3),...
        "MaxEpoch", 200,...
        "BatchNormType", "none",...
        "LearningRate", 0.1,...
        "SolverName", "adam",...
        "WeightInitializationType", "none",...
        "DataNormalize" , "X",...
        "MiniBatchSize", 128,...
        "adampar_beta1", 0.6,...
        "adampar_beta2", 0.8,...
        "adampar_epsilon", 1e-8,...
        "adampar_m0", 0,...
        "adampar_v0", 0,...
        "Plot", 1,...
        "Verbose", 10, ...
        "MultiClassMode","softmax")
    % net.lamReg = 10;

    %%
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
    %%
    for i = 2:n_run
        i
        tic
        %%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%
        [trained_net,valdata] = net.Train(...
            "validationSplitPercent",0.2,...
            "valPerEpochFrequency",1,...
            "ApplyRuleRemover",0);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        texe(i) = toc;
        [~,metricsLast] = Test(trained_net.last, TE.x, TE.y);
        [~,metricsBest] = Test(trained_net.best, TE.x, TE.y);
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
        [~,m_b] = Test(TRDnet, TE.x, TE.y, "Plot",0,"MetricReport","all");
        [~,m_p] = Test(net_removedRule.Percentage, TE.x, TE.y, "Plot",0,"MetricReport","all");
        [~,m_mus] = Test(net_removedRule.MeanMultiStd, TE.x, TE.y, "Plot",0,"MetricReport","all");
        [~,m_mis] = Test(net_removedRule.MeanMinesStd, TE.x, TE.y, "Plot",0,"MetricReport","all");
        [~,m_n] = Test(net_removedRule.new, TE.x, TE.y, "Plot",0,"MetricReport","all");
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
    %%%%%%%%%%%%%%%%%%%% save to exel %%%%%%%%%%%%%%%%%%%%%
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
    end % num test
    writetable(tbl, excelFileName, 'Sheet', datasetName);
end % dataset

%%
% net_removedRule = RuleRemover(trained_net.best,Xval,0.5,0.5,...
%     "OptimizeParams",0,...
%     "OutputData_required",Yval,...
%     "DisplayIterations","iter",...
%     "PSO_MaxIterations",20,...
%     "PSO_MaxStallIterations",7)
% 
% 
% % Test
% figure
% plot(Yte,DisplayName='Ref')
% hold on
% networks = [trained_net.last, ...
%     trained_net.best, ...
%     net_removedRule.Percentage, ...
%     net_removedRule.MeanMultiStd, ...
%     net_removedRule.MeanMinesStd, ...
%     net_removedRule.new];
% name = ["last","best","perc","mean*std","mean-std","new"];
% markers = ["*","o","+","square",">","x","."];
% for i = 1:numel(networks)
%     [yhat,err] = Test(networks(i), Xte, Yte, "Plot",0,"MetricReport","all");
%     disp(err)
%     disp(networks(i).n_rulePerLayer)
%     plot(yhat,DisplayName=name(i),Marker=markers(i),MarkerIndices=1:10:numel(Yte),LineWidth=1.5)
% end
% legend('show')

