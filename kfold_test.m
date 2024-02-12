clear
clc
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excelFileName = 'results_kfold.xlsx';
datasetDirectory = "./dataset";
kfold = 4;

% for datasetName = ["sp500", "mglass", "ampg"] % regression
% for datasetName = ["pen", "gesphase", "liver", "kmg"] % classification
sheetName = ["1"];
datasetName = ["liver"];
[TR,TE] = datasetBenchmark(datasetName,datasetDirectory,kfold);
param = [1.5 1 0.1 0.001];
for j = 1:1
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % [TR,TE] = datasetBenchmark(datasetName(j),datasetDirectory,kfold);

    %%
    clc
    for str = ["m_b","m_p","m_mus","m_mis","m_n"]
        eval(str+"_LOSS = zeros("+kfold+",1);")
        eval(str+"_MSE = zeros("+kfold+",1);")
        eval(str+"_RMSE = zeros("+kfold+",1);")
        eval(str+"_MAE = zeros("+kfold+",1);")
        eval(str+"_NDEI = zeros("+kfold+",1);")
        eval(str+"_NDEI2 = zeros("+kfold+",1);")
        eval(str+"_ACC = zeros("+kfold+",1);")
        eval(str+"_BACC = zeros("+kfold+",1);")
        eval(str+"_PREC = zeros("+kfold+",1);")
        eval(str+"_RECALL = zeros("+kfold+",1);")
        eval(str+"_F1SCORE = zeros("+kfold+",1);")
        eval(str+"_KAPPA = zeros("+kfold+",1);")
        eval(str+"_MCC = zeros("+kfold+",1);")
    end
    [texe,layers,rules,rem_rule_p,rem_rule_mus,rem_rule_mis,rem_rule_n]= deal(zeros(kfold,1));
    %%
    for fold = 1:kfold
        fold

        % construct neuro-fuzzy network
        close all
        % rng(42)
        net = SOMFNN(TR{fold}.x,TR{fold}.y,2,...
            "n_hiddenNodes","auto",... [vector,"auto"]
            "ActivationFunction", ["sig"],... [rel,leaky,sig,tan,lin,elu]
            "KernelsName",["RBF"],... ["RBF","COS","Linear","Polynomial","Laplacian","Chi2"]
            "KernelsWeight",[1], ... sum = 1
            "DensityThreshold", exp(-3),...
            "MaxEpoch", 200,...
            "LearningRateSchedule","none",... ["none",[epoch,LR,...,epoch,LR]]
            "BatchNormType", "none",... temp
            "LearningRate", 0.01,...
            "SolverName", "adam",... [mini,sgd,adam]
            "WeightInitializationType", "mean",... [mean,rand,xavier]
            "DataNormalize" , "X",...
            "NormalizeDimension","eachFeature",... ["eachFeature","dataFeatures"]
            "MiniBatchSize", 128,...
            "adampar_beta1", 0.6,...
            "adampar_beta2", 0.8,...
            "adampar_epsilon", 1e-8,...
            "adampar_m0", 0,...
            "adampar_v0", 0,...
            "Plot", 1,... [0,1]
            "Verbose", 50, ... [0:MaxEpoch)
            "RegularizationTerm",param(fold), ...[0:inf)
            "updatePrototype",0 ...[0,1]
            )


        
        %%%%%%%%%%%%%%% TRAIN + VAL %%%%%%%%%%%%%%%%%%%%%%%%
        tic
        [trained_net,valdata] = net.Train( ...
            "Xval",[],...
            "Yval",[],...
            "validationSplitPercent",0.1,... [0:1)
            "valPerEpochFrequency",1,... [0:MaxEpoch)
            "ApplyRuleRemover",0, ... [0,1]
            "LiveOutPlot","none", ... ["none",dataTest,"useTrainData","useValData"]
            "LiveOutPlotFrequency",5, ... [0:MaxEpoch)
            "EarlyStoppingPatience",inf ... [0:MaxEpoch)
            );
        texe(fold) = toc;
        [~,metricsLast] = Test(trained_net.last, TE{fold}.x, TE{fold}.y);
        [~,metricsBest] = Test(trained_net.best, TE{fold}.x, TE{fold}.y);
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
        %%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%
        % tic
        % [TRDnet,valdata] = net.Train( ...
        %     "Xval",[],...
        %     "Yval",[],...
        %     "validationSplitPercent",0,... [0:1)
        %     "valPerEpochFrequency",1,... [0:MaxEpoch)
        %     "ApplyRuleRemover",0, ... [0,1]
        %     "LiveOutPlot","none", ... ["none",dataTest,"useTrainData","useValData"]
        %     "LiveOutPlotFrequency",5 ...
        %     )
        % texe(fold) = toc;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        net_removedRule = RuleRemover(TRDnet,valdata.x,0.5,0.5,...
            "OptimizeParams",0,...
            "OutputData_required",[],...
            "DisplayIterations","iter",...
            "PSO_MaxIterations",20,...
            "PSO_MaxStallIterations",7);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        layers(fold) = TRDnet.n_Layer;
        rules(fold) = mean(TRDnet.n_rulePerLayer);
        rem_rule_p(fold) =  mean(net_removedRule.Percentage.n_rulePerLayer);
        rem_rule_mus(fold) =  mean(net_removedRule.MeanMultiStd.n_rulePerLayer);
        rem_rule_mis(fold) =  mean(net_removedRule.MeanMinesStd.n_rulePerLayer);
        rem_rule_n(fold) =  mean(net_removedRule.new.n_rulePerLayer);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [~,m_b] = Test(TRDnet, TE{fold}.x, TE{fold}.y, "Plot",0,"MetricReport","all");
        [~,m_p] = Test(net_removedRule.Percentage, TE{fold}.x, TE{fold}.y, "Plot",0,"MetricReport","all");
        [~,m_mus] = Test(net_removedRule.MeanMultiStd, TE{fold}.x, TE{fold}.y, "Plot",0,"MetricReport","all");
        [~,m_mis] = Test(net_removedRule.MeanMinesStd, TE{fold}.x, TE{fold}.y, "Plot",0,"MetricReport","all");
        [~,m_n] = Test(net_removedRule.new, TE{fold}.x, TE{fold}.y, "Plot",0,"MetricReport","all");
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for str = ["m_b","m_p","m_mus","m_mis","m_n"]
            if contains(net.ProblemType,"Regression")
                eval(str+"_LOSS(fold) = "+str+".LOSS;")
                eval(str+"_MSE(fold) = "+str+".MSE;")
                eval(str+"_RMSE(fold) = "+str+".RMSE;")
                eval(str+"_MAE(fold) = "+str+".MAE;")
                eval(str+"_NDEI(fold) = "+str+".NDEI;")
                eval(str+"_NDEI2(fold) = "+str+".NDEI2;")
            else
                eval(str+"_ACC(fold) = "+str+".ACC;")
                eval(str+"_BACC(fold) = "+str+".BACC;")
                eval(str+"_PREC(fold) = "+str+".PREC;")
                eval(str+"_RECALL(fold) = "+str+".RECALL;")
                eval(str+"_F1SCORE(fold) = "+str+".F1SCORE;")
                eval(str+"_KAPPA(fold) = "+str+".KAPPA;")
                eval(str+"_MCC(fold) = "+str+".MCC;")
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%% save to exel %%%%%%%%%%%%%%%%%%%%%
    par = (1:kfold)';
    % par = param';
    if contains(net.ProblemType,"Regression")
        tbl = table(par,texe,layers, ...
            rules,m_b_LOSS,m_b_MSE,m_b_RMSE,m_b_MAE,m_b_NDEI,m_b_NDEI2, ...
            rem_rule_p,m_p_LOSS,m_p_MSE,m_p_RMSE,m_p_MAE,m_p_NDEI,m_p_NDEI2, ...
            rem_rule_mus,m_mus_LOSS,m_mus_MSE,m_mus_RMSE,m_mus_MAE,m_mus_NDEI,m_mus_NDEI2, ...
            rem_rule_mis,m_mis_LOSS,m_mis_MSE,m_mis_RMSE,m_mis_MAE,m_mis_NDEI,m_mis_NDEI2, ...
            rem_rule_n,m_n_LOSS,m_n_MSE,m_n_RMSE,m_n_MAE,m_n_NDEI,m_n_NDEI2);
    else
        tbl = table(par,texe,layers, ...
            rules,m_b_ACC,m_b_BACC,m_b_PREC,m_b_RECALL,m_b_F1SCORE,m_b_KAPPA,m_b_MCC, ...
            rem_rule_p,m_p_ACC,m_p_BACC,m_p_PREC,m_p_RECALL,m_p_F1SCORE,m_p_KAPPA,m_p_MCC, ...
            rem_rule_mus,m_mus_ACC,m_mus_BACC,m_mus_PREC,m_mus_RECALL,m_mus_F1SCORE,m_mus_KAPPA,m_mus_MCC, ...
            rem_rule_mis,m_mis_ACC,m_mis_BACC,m_mis_PREC,m_mis_RECALL,m_mis_F1SCORE,m_mis_KAPPA,m_mis_MCC, ...
            rem_rule_n,m_n_ACC,m_n_BACC,m_n_PREC,m_n_RECALL,m_n_F1SCORE,m_n_KAPPA,m_n_MCC);
    end % num test
    writetable(tbl, excelFileName, 'Sheet', sheetName(j));

end % dataset