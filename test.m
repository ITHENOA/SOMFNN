clear
clc
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excelFileName = 'results2.xlsx';
datasetDirectory = "./dataset";
[TR,TE] = datasetBenchmark("liver",datasetDirectory);
n_run = 10;
% for datasetName = ["sp500", "mglass", "ampg"] % regression
% for datasetName = ["pen", "gesphase", "liver", "kmg"] % classification

datasetName = ["liver_RT=0"];
% par = [0 0.001 1.5]
for j = 1:1%numel(datasetName)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % [TR,TE] = datasetBenchmark(datasetName,datasetDirectory);

    % construct neuro-fuzzy network
    clc,close all
    % rng(42)
    net = SOMFNN(TR.x,TR.y,2,...
        "n_hiddenNodes","auto",... [vector,"auto"]
        "ActivationFunction", ["sig"],... [rel,leaky,sig,tan,lin,elu]
        "KernelsName",["RBF"],... ["RBF","COS","Linear","Polynomial","Laplacian","Chi2"]
        "KernelsWeight",[1], ... sum = 1
        "DensityThreshold", exp(-3),...
        "MaxEpoch", 200,...
        "LearningRateSchedule","none",... ["none",[epoch,LR,...,epoch,LR]]
        "BatchNormType", "none",... temp
        "LearningRate", 1,...
        "SolverName", "adam",... [mini,sgd,adam,batchgd,batchadam]
        "WeightInitializationType", "mean",... [mean,rand,xavier]
        "DataNormalize" , "X",...
        "NormalizeDimension","eachFeature",... ["eachFeature","dataFeatures"]
        "MiniBatchSize", 201,...
        "adampar_beta1", .6,...
        "adampar_beta2", .8,...
        "adampar_epsilon", 1e-8,...
        "adampar_m0", 0,...
        "adampar_v0", 0,...
        "Plot", 1,... [0,1]
        "Verbose", 10, ... [0:MaxEpoch)
        "RegularizationTerm",0, ...[0:inf)
        "updatePrototype",0 ...[0,1]
        )

    %%
    for str = ["m_b","m_p","m_mus","m_mis","m_n"]
        eval(str+"_LOSS = zeros("+n_run+",1);")
        eval(str+"_MSE = zeros("+n_run+",1);")
        eval(str+"_RMSE = zeros("+n_run+",1);")
        eval(str+"_MAE = zeros("+n_run+",1);")
        eval(str+"_NDEI = zeros("+n_run+",1);")
        eval(str+"_NDEI2 = zeros("+n_run+",1);")
        eval(str+"_ACC = zeros("+n_run+",1);")
        eval(str+"_BACC = zeros("+n_run+",1);")
        eval(str+"_PREC = zeros("+n_run+",1);")
        eval(str+"_RECALL = zeros("+n_run+",1);")
        eval(str+"_F1SCORE = zeros("+n_run+",1);")
        eval(str+"_KAPPA = zeros("+n_run+",1);")
        eval(str+"_MCC = zeros("+n_run+",1);")
    end
    [texe,layers,rules,rem_rule_p,rem_rule_mus,rem_rule_mis,rem_rule_n]= deal(zeros(n_run,1));
    %%
    for i = 1:n_run
        i
        
        %%%%%%%%%%%%%%% TRAIN + VAL %%%%%%%%%%%%%%%%%%%%%%%%
        tic
        [trained_net,valdata] = net.Train( ...
            "Xval",[],... 
            "Yval",[],...
            "validationSplitPercent",0.2,... [0:1)
            "valPerEpochFrequency",1,... [0:MaxEpoch)
            "ApplyRuleRemover",0, ... [0,1]
            "LiveOutPlot","none", ... ["none",dataTest,"useTrainData","useValData"]
            "LiveOutPlotFrequency",5, ...
            "EarlyStoppingPatience",1000 ... [0:MaxEpoch)
            );
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
        %%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%
        % tic
        % [TRDnet,valdata] = net.Train( ...
        %     "Xval",[],... 
        %     "Yval",[],...
        %     "validationSplitPercent",0.2,... [0:1)
        %     "valPerEpochFrequency",1,... [0:MaxEpoch)
        %     "ApplyRuleRemover",0, ... [0,1]
        %     "LiveOutPlot","none", ... ["none",dataTest,"useTrainData","useValData"]
        %     "LiveOutPlotFrequency",5 ...
        %     );
        % texe(i) = toc;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        net_removedRule = RuleRemover(TRDnet,valdata.x,0.5,0.5,...
            "OptimizeParams",0,...
            "OutputData_required",valdata.y,...
            "DisplayIterations","iter",...
            "PSO_MaxIterations",1000,...
            "PSO_MaxStallIterations",100);
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
                eval(str+"_BACC(i) = "+str+".BACC;")
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
            rules,m_b_ACC,m_b_BACC,m_b_PREC,m_b_RECALL,m_b_F1SCORE,m_b_KAPPA,m_b_MCC, ...
            rem_rule_p,m_p_ACC,m_p_BACC,m_p_PREC,m_p_RECALL,m_p_F1SCORE,m_p_KAPPA,m_p_MCC, ...
            rem_rule_mus,m_mus_ACC,m_mus_BACC,m_mus_PREC,m_mus_RECALL,m_mus_F1SCORE,m_mus_KAPPA,m_mus_MCC, ...
            rem_rule_mis,m_mis_ACC,m_mis_BACC,m_mis_PREC,m_mis_RECALL,m_mis_F1SCORE,m_mis_KAPPA,m_mis_MCC, ...
            rem_rule_n,m_n_ACC,m_n_BACC,m_n_PREC,m_n_RECALL,m_n_F1SCORE,m_n_KAPPA,m_n_MCC);
    end % num test
    writetable(tbl, excelFileName, 'Sheet', datasetName(j));
end % dataset