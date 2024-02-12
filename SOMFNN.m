% MB : Mini Batch
% k : Instance
% l : layer
% n : rule/cluster
% X : xbar : [1;k]
classdef SOMFNN
    properties
        n_Layer
        n_rulePerLayer
        n_nodes
        DensityThreshold
        LearningRate
        MiniBatchSize
        MaxEpoch
        TrainedEpoch = 0
        Layer
        SolverName
        ActivationFunction
        WeightInitializationType
        DataNormalize
        BatchNormType
        MSE_report
        Verbose
        Plot
        ProblemType
        RegularizationTerm
        updatePrototype
        KernelsName
        KernelsWeight
        nTrData
        LearningRateSchedule
    end
    properties (Access=private)
        NormalizeDimension
        Xtrain
        Ytrain
        % n_featureX
        % n_featureY
        dataSeenCounter
        % lambda_MB                   % {layer}(rule,MB)
        % AF_AX                       % AF(A*xbar) : {layer}(Wl,1)
        % DlamDx                      % {layer}(Ml,1)
        % AFp_AX                      % derivative of AF : {layer}()
        adapar                      % parameters of Adam algorithm
        classification_flag = 0     % 1:binery, 2:multiclass
        regression_flag = 0         % 1:sequence-to-one(MISO), 2:Sequence-to-sequence(MIMO)
        validation_flag = 0         % 1:using validation
        minY
        maxY
        uniqLabels
        LRScheduler_flag = 0
        ini_LR
    end
    methods
        %% ----------------------- construct ------------------------------
        function o = SOMFNN(Xtr, Ytr, n_Layer, opts)
            arguments
                Xtr % (#data,#dim)                : Input data
                Ytr % (#data,#dim)                : Output data
                n_Layer (1,1) {mustBeInteger,mustBePositive}
                opts.n_hiddenNodes (1,:) = "Auto"
                opts.LearningRate (1,1) {mustBePositive} = 0.001
                opts.MaxEpoch (1,1) {mustBeInteger,mustBePositive} = 500
                opts.DensityThreshold (1,1) {mustBePositive} = exp(-3)
                opts.verbose (1,1) {mustBeInteger} = 0
                opts.plot (1,1) {logical} = 0
                opts.ActivationFunction = "Sigmoid"
                opts.BatchNormType {mustBeTextScalar} = "none"
                opts.SolverName {mustBeTextScalar} = "SGD"
                opts.WeightInitializationType {mustBeTextScalar} = "rand"
                opts.MiniBatchSize {mustBeInteger,mustBePositive} = 1
                opts.DataNormalize = "none"
                opts.adampar_epsilon = 1e-8
                opts.adampar_beta1 = 0.9
                opts.adampar_beta2 = 0.999
                opts.adampar_m0 = 0
                opts.adampar_v0 = 0
                opts.RegularizationTerm = 0
                opts.updatePrototype = 0
                opts.KernelsName {isstring,isvector} = "RBF"
                opts.KernelsWeight {isnumeric,isvector} = 1
                opts.NormalizeDimension {mustBeTextScalar} = "eachFeature" % ["eachFeature","dataFeatres"]
                opts.LearningRateSchedule = "none" % ["none",[epoch,LR,...,epoch,LR]]
            end

            [Xtr, Ytr] = o.checkData(Xtr, Ytr);

            % Save to object
            o.LearningRate = opts.LearningRate;
            o.MaxEpoch = opts.MaxEpoch;
            o.DensityThreshold = opts.DensityThreshold;
            o.MiniBatchSize = opts.MiniBatchSize;
            o.Verbose = opts.verbose;
            o.Plot = opts.plot;
            o.n_Layer = n_Layer;
            o.ActivationFunction = opts.ActivationFunction;
            o.WeightInitializationType = opts.WeightInitializationType;
            o.BatchNormType = opts.BatchNormType;
            o.DataNormalize = opts.DataNormalize;
            o.SolverName = opts.SolverName;
            o.RegularizationTerm = opts.RegularizationTerm;
            o.updatePrototype = opts.updatePrototype;
            o.KernelsName = opts.KernelsName;
            o.KernelsWeight = opts.KernelsWeight;
            o.NormalizeDimension = opts.NormalizeDimension;
            o.LearningRateSchedule = opts.LearningRateSchedule;
            o.ini_LR = o.LearningRate;

            % Validate String Variables
            o = o.StringValidate();

            % check KernelsWeight
            if numel(o.KernelsWeight) ~= numel(o.KernelsName)
                error("number of elemetns {'KernelsName','KernelsWeight'} must be same.")
            end
            % if sum(o.KernelsWeight) ~= 1
            %     warning("'KernelsWeight' must be probability (sum=1), I will fix it.")
            %     o.KernelsWeight = o.KernelsWeight/sum(o.KernelsWeight);
            % end

            if ~strcmp(o.LearningRateSchedule,"none") && isnumeric(o.LearningRateSchedule)
                o.LRScheduler_flag = 1;
            end


            n_node_end = size(Ytr,2);
            % Problem Type
            uniqOutputs = unique(Ytr);
            if all(mod(Ytr, 1) == 0) % Classification
                n_class = numel(uniqOutputs);
                o.uniqLabels = uniqOutputs;
                if n_class == 2 %all(ismember(uniqOutputs, [0, 1]))
                    disp('Problem Type: Binary Classification');
                    o.ProblemType = "Binary-Classification";
                    o.classification_flag = 1;
                    if ismember(2,uniqOutputs), error("class labels should be [0,1]."), end
                    if ~strcmp(o.ActivationFunction(end),"Sigmoid")
                        warning("Activaion function of last layer for binary classification should be 'Sigmoid', I will fix it.")
                        o.ActivationFunction(end) = "Sigmoid";
                    end
                elseif n_class > 2
                    disp('Problem Type: Multiclass Classification');
                    o.ProblemType = "Multiclass-Classification";
                    o.classification_flag = 2;
                    if min(uniqOutputs) == 0, Ytr = Ytr + 1; end % label classes 1,2,...
                    n_node_end = n_class;
                    if ~strcmp(o.ActivationFunction(end),"Linear")
                        warning("Activaion function of last layer for multiclass classification should be 'Linear', I will fix it.")
                        o.ActivationFunction(end) = "Linear";
                    end
                end
            else % Regression
                if size(Ytr,2) == 1
                    disp('Problem Type: MISO Regression');
                    o.ProblemType = "MISO-Regression";
                    o.regression_flag = 1;
                else
                    disp('Problem Type: MIMO Regression');
                    o.ProblemType = "MIMO-Regression";
                    o.regression_flag = 2;
                end
            end

            % n_nodes
            if ~isnumeric(opts.n_hiddenNodes)
                opts.n_hiddenNodes = ones(1,n_Layer-1) * 3 * n_node_end;
            elseif numel(opts.n_hiddenNodes) == 1
                opts.n_hiddenNodes = ones(1,n_Layer-1) * opts.n_hiddenNodes;
            elseif numel(opts.n_hiddenNodes) ~= n_Layer-1
                error("incorrect number of vector 'n_hiddenNodes'.")
            end
            W = [opts.n_hiddenNodes,n_node_end];
            M = [size(Xtr,2),opts.n_hiddenNodes];
            if sum(M < W)
                [~,l] = find(M < W);
                warning(['Number of outputs of layer (%s) may cause overfitting \n' ...
                    'Fix this: (W(%d)=%d <= %d) \nHint: W(l) <= M(l)'], ...
                    num2str(l), l(1), W(l(1)), M(l(1)));
            end
            if sum(W < W(end))
                [~,l] = find(W < W(end));
                warning(['Number of outputs of layer (%s) may cause lossing too much informations \n' ...
                    'Fix this: (W(%d)=%d >= %d) \nHint: W(l) >= output dimension'], ...
                    num2str(l), l(1), W(l(1)), W(end));
            end
            o.n_nodes = [M,W(end)];

            % Adam Parameters
            if sum(strcmp(o.SolverName,["Adam","BatchAdam"]))
                % Adam Parameters
                o.adapar.ini_m = opts.adampar_m0;
                o.adapar.ini_v = opts.adampar_v0;
                o.adapar.b1 = opts.adampar_beta1;
                o.adapar.b2 = opts.adampar_beta2;
                o.adapar.epsilon = opts.adampar_epsilon;
                % Prepare Adam Parameters
                o.adapar.m = cell(1,o.n_Layer);
                o.adapar.v = cell(1,o.n_Layer);
                [o.adapar.m{:}] = deal(o.adapar.ini_m);
                [o.adapar.v{:}] = deal(o.adapar.ini_v);
            elseif strcmp(o.SolverName,"SGD") && o.MiniBatchSize > 1
                warning("In SGD Algorithm batch_size should be 1; I fixed this for you")
                o.MiniBatchSize = 1;
            end

            % initialize variables
            o = o.ConstructVars();

            % save data
            o.Xtrain = Xtr;
            o.Ytrain = Ytr;
            o.minY = min(Ytr);
            o.maxY = max(Ytr);
        end

        %% ------------------------- TRAIN -------------------------
        function [trained_net,valData] = Train(o,opts)
            arguments
                o
                % or
                opts.Xval = []
                opts.Yval = []
                % or
                opts.validationSplitPercent = 0
                % optional
                opts.valPerEpochFrequency = 1 % test val every 1 epochs
                opts.ApplyRuleRemover {logical} = 0
                opts.LiveOutPlot = "none" % [defult="none", dataTest, "useTrainData", "useValData"]
                opts.LiveOutPlotFrequency = 5
                opts.EarlyStoppingPatience {mustBeNumeric} = inf
            end
            % Error check
            if ~opts.validationSplitPercent && opts.ApplyRuleRemover
                warning("Validation data not exist, Rule-Remover will work with training data, which it may havent good performance.")
            end
            if opts.validationSplitPercent && (exist("opts.Xval","var") || exist("opts.Yval","var"))
                if (exist("opts.Xval","var") && exist("opts.Yval","var"))
                    warning("Validation data entered, so set 'valPerEpochFrequency' to zero.");
                else
                    error("Enter both of Xval and Yval")
                end
            end
            
            o = o.StringValidate();
            if o.Plot, fPlot = figure(Name="Process",NumberTitle="off"); end
            if opts.validationSplitPercent || ~isempty(opts.Xval)
                o.validation_flag = 1;
            end
            valData = [];
            if o.validation_flag % Validation
                if ~isempty(opts.Xval)
                    Xval = opts.Xval;
                    Yval = opts.Yval;
                    % n_val = size(Xval,1);
                else
                    idx = randperm(size(o.Xtrain,1));
                    n_val = round(size(o.Xtrain,1) * opts.validationSplitPercent);
                    Xval = o.Xtrain(idx(1:n_val),:);
                    Yval = o.Ytrain(idx(1:n_val),:);
                    o.Xtrain = o.Xtrain(idx(n_val+1:end),:);
                    o.Ytrain = o.Ytrain(idx(n_val+1:end),:);
                end
                bestValLoss = inf;
                valEpochs = opts.valPerEpochFrequency:opts.valPerEpochFrequency:o.MaxEpoch;
                METRIC_val = zeros(1,numel(valEpochs));
                LOSS_val = zeros(1,numel(valEpochs));
                valCount = 0;
                valStall = 0;
            end
            o.nTrData = size(o.Xtrain,1);
            
            % normalize data if needed
            [Xtr,Ytr] = o.NormalizeData(o.Xtrain,o.Ytrain);

            if sum(strcmp(o.SolverName,["BatchGD","BatchAdam"]))
                o.MiniBatchSize = size(Xtr,1);
            end
            % LiveOutPlot
            if ~strcmp(opts.LiveOutPlot,"none")
                if strcmp(opts.LiveOutPlot,"useTrainData")
                    LiveOutPlotX =  Xtr;
                    LiveOutPlotY = Ytr;
                elseif strcmp(opts.LiveOutPlot,"useValData")
                    if ~o.validation_flag, error("use validation data"),end
                    LiveOutPlotX =  Xval;
                    LiveOutPlotY = Yval;
                elseif isnumeric(opts.LiveOutPlot)
                    LiveOutPlotX =  opts.LiveOutPlot(:,1:end-1);
                    LiveOutPlotY = opts.LiveOutPlot(:,end);
                    if o.classification_flag == 2 && min(unique(LiveOutPlotY)) == 0
                        LiveOutPlotY = LiveOutPlotY + 1;
                    end
                else, error("Invalid 'LiveOutPlot' option name.")
                end
                opts.LiveOutPlot = 1;
                fLive = figure(Name="Live Output with user data entered",NumberTitle="off");
            else
                opts.LiveOutPlot = 0;
            end

            it = 0; % iteration total
            METRIC_tr = zeros(1,o.MaxEpoch);
            LOSS_tr = zeros(1,o.MaxEpoch);
            o.dataSeenCounter = 0;

            %%%%%%%% epoch %%%%%%%%
            for epoch = 1:o.MaxEpoch


                % % Learning Rate Decay
                % if o.LearningRateDecayFactor
                    % o.LearningRate = (1 / (1 + 0.01 * epoch)) * o.ini_LR;
                    % o.LearningRate = (0.95 ^ epoch) * o.ini_LR;
                % end
                % Schedule Learning Rate
                if o.LRScheduler_flag && ismember(epoch,o.LearningRateSchedule(1:2:end))
                    o.LearningRate = o.LearningRateSchedule(find(epoch == o.LearningRateSchedule(1:2:end))*2);
                end

                yhat = zeros(o.nTrData,o.n_nodes(end));
                shuffle_idx = randperm(size(Xtr,1));
                %%%%%%%%%%% iteration %%%%%%%%%%%
                for it_epoch = 1 : ceil(size(Xtr,1)/o.MiniBatchSize)
                    it = it + 1;
                    MB_idx = shuffle_idx( (it_epoch-1)*o.MiniBatchSize+1 : min(it_epoch*o.MiniBatchSize,size(Xtr,1)) );

                    x = Xtr(MB_idx,:)';
                    if ~strcmp(o.BatchNormType,"none")
                        x = normalize(x,1,o.BatchNormType); % dim=2; normalize each feature
                    end
                    y = Ytr(MB_idx,:)';

                    % Forward >> Backward
                    [o,yhat_MB] = o.Main(x,y,it);

                    % save pars
                    yhat(MB_idx,:) = yhat_MB'; % error per it
                    o.dataSeenCounter = o.dataSeenCounter + numel(MB_idx);
                end %%%%%%%%%%% END iteration %%%%%%%%%%%

                %%%%%%%%%%% epoch - results %%%%%%%%%%%

                % un normalize data (if needed)
                yhat = o.UnNormalizeOutput(yhat);

                % calculate loss
                LOSS_tr(epoch) = o.GetLoss(yhat,o.Ytrain);

                % class label detection (if needed)
                if o.classification_flag
                    yhat = o.GetClassLabel(yhat);
                end

                % calculate metrices
                METRIC_tr(epoch) = o.GetMetric(yhat,o.Ytrain); % MSE/ACC for regression/classification

                % verbose
                if o.Verbose
                    if ~rem(epoch,o.Verbose)
                        if o.regression_flag
                            fprintf("[Epoch:%d] [MSE:%.4f] [RMSE:%.4f] \n", epoch, METRIC_tr(epoch), sqrt(METRIC_tr(epoch)))
                        else
                            fprintf("[Epoch:%d] [ACC:%.3f] [Loss:%.4f] \n", epoch, METRIC_tr(epoch), LOSS_tr(epoch))
                        end
                    end
                end

                % validation
                if o.validation_flag && ismember(epoch,valEpochs)
                    valCount = valCount + 1;
                    [~,metric_val] = o.Test(Xval,Yval);
                    METRIC_val(valCount) = metric_val.MSEorACC; % (reg=MSE),(clas=ACC)
                    LOSS_val(valCount) = metric_val.LOSS;
                    if valCount>1
                        if (LOSS_val(valCount) < bestValLoss)
                            netBest = o;
                            bestValLoss = LOSS_val(valCount);
                            valStall = 0;
                            % o.RegularizationTerm = o.RegularizationTerm * 1.01;
                        else
                            valStall = valStall + 1;
                            if valStall > opts.EarlyStoppingPatience, break, end
                            % if valStall > 5, o.RegularizationTerm = o.RegularizationTerm*1.01;end
                            % o.RegularizationTerm
                            % if LOSS_tr(epoch) < min(LOSS_tr(1:epoch-1))
                            % o.RegularizationTerm = max(o.RegularizationTerm * 0.9,0.001);
                            % if o.RegularizationTerm < 0.0001
                            %     o.RegularizationTerm = 0.01;
                            % end
                            % o.RegularizationTerm
                            % end
    
                        end
                    end
                    % if valCount > 40 && LOSS_val(valCount) > LOSS_val(valCount-40)
                    %     o.RegularizationTerm = 0.01;
                    %     o.RegularizationTerm
                    % end
                end

                o.TrainedEpoch = epoch;

                % LiveOutPlot
                if opts.LiveOutPlot && ~rem(epoch,opts.LiveOutPlotFrequency)
                    [yhLive,~] = o.Test(LiveOutPlotX);
                    figure(fLive),hold on
                    if o.regression_flag
                        plot(1:numel(LiveOutPlotY),LiveOutPlotY)
                        plot(1:numel(LiveOutPlotY),yhLive)
                        legend("Refrence","Predicted")
                    else
                        plotconfusion(categorical(LiveOutPlotY),categorical(yhLive))
                    end
                    drawnow,hold off
                end

                % Plot
                if (o.Plot) && (epoch > 1)
                    try figure(fPlot),catch,end
                    if o.classification_flag
                        if o.validation_flag
                            o.GetPlot(METRIC_tr(1:epoch)*100, LOSS_tr(1:epoch), METRIC_val(1:valCount)*100, LOSS_val(1:valCount), valEpochs(1:valCount), ["Acuracy","Loss"])
                        else
                            o.GetPlot(METRIC_tr(1:epoch)*100, LOSS_tr(1:epoch), [], [], [], ["Acuracy","Loss"])
                        end
                    elseif o.regression_flag
                        if o.validation_flag
                            o.GetPlot(METRIC_tr(1:epoch), sqrt(METRIC_tr(1:epoch)), METRIC_val(1:valCount), sqrt(METRIC_val(1:valCount)), valEpochs(1:valCount), ["MSE","RMSE"])
                        else
                            o.GetPlot(METRIC_tr(1:epoch), sqrt(METRIC_tr(1:epoch)), [], [], [], ["MSE","RMSE"])
                        end
                    end
                end

                % Eliminate Condition
                if (epoch>5) && prod(METRIC_tr(epoch-5:end) - METRIC_tr(epoch)), break, end
            end %%%%%%%%%%% END EPOCH %%%%%%%%%%%

            % Apply Rule Remover (if needed)
            if opts.ApplyRuleRemover
                if o.validation_flag && epoch>1
                    o = o.RuleRemover(Xval,0.5,0.5);
                    netBest = netBest.RuleRemover(Xval,0.5,0.5);
                else
                    o = o.RuleRemover(o.Xtrain,0.5,0.5);
                end
            end

            % save network
            if o.validation_flag && epoch>1
                trained_net.last = o;
                trained_net.best = netBest;
                valData.x = Xval;
                valData.y = Yval;
            else
                trained_net = o;
            end

            % return results
            % [bestMSE,bestIdx] = min(METRIC_tr);
            % meanMSE = mean(METRIC_tr);
            % o.MSE_report.Mean = meanMSE;
            % o.MSE_report.Best = bestMSE;
            % o.MSE_report.Last = METRIC_tr(end);
            %
            % METRIC_tr = ["Last";"Best";"Mean"];
            % Value = [METRIC_tr(end); bestMSE; meanMSE];
            % Epoch = [epoch; bestIdx; nan];
            % table(METRIC_tr,Value,Epoch)
            % o.MSE_report = sprintf("[Mean:%.3f, Best:%.3f]",mean(MSE_ep),min(MSE_ep));
            
        end

        %% ----------------------------- TEST -----------------------------
        function [yhat, metrics, lambda] = Test(net,Xtest,Ytest,opts)
            arguments
                net
                Xtest
                Ytest = []
                opts.MetricReport {mustBeTextScalar} = "none" % ["none", "all"]
                opts.Plot {logical} = 0
            end
            [Xtest,Ytest] = net.checkData(Xtest,Ytest);
            metrics = [];
            lambda = cell(1,net.n_Layer);

            Yexist = 1;
            if isempty(Ytest), Yexist = 0; end
            net = net.StringValidate();

            % normalize data if needed
            Xtest = net.NormalizeData(Xtest);

            x = Xtest';
            % Forward
            for l = 1:net.n_Layer
                lambda{l} =  net.GetLambda(x, net.Layer{l}, 1:net.n_rulePerLayer(l));
                if isnan(sum(lambda{l},"all"))
                    error
                end
                x = net.GetOutput(net.Layer{l}, x, lambda{l});
            end
            % un normalize if needed
            yhat = net.UnNormalizeOutput(x');

            % check labels
            if net.classification_flag == 2 ...
                    && min(net.uniqLabels) == 0 ...
                    && min(unique(Ytest)) == 0 ...
                    && Yexist
                Ytest = Ytest + 1; 
            end

            % determin loss (required Ytest)
            if Yexist, Loss = net.GetLoss(yhat,Ytest); end

            % prepare output for classification problem
            if net.classification_flag,  yhat = net.GetClassLabel(yhat); end

            if Yexist
                % METRIC
                metrics = net.GetMetric(yhat,Ytest,opts.MetricReport);
                if ~isstruct(metrics)
                    metrics = struct('MSEorACC',metrics);
                    metrics.LOSS = Loss;
                else
                    metrics.LOSS = Loss;
                end
                % PLOT
                if opts.Plot
                    figure
                    if net.regression_flag
                        plot(Ytest), hold on, plot(yhat)
                    elseif net.classification_flag
                        % confusionchart(Ytest,yhat)
                        plotconfusion(categorical(Ytest),categorical(yhat))
                    end
                end
            end
        end

        %%  ----------------------------- main -----------------------------
        function [o,yhat] = Main(o,x,y,it)
            % forward >> Create Rules and estimaye final output
            [yhat,o,BP_pars] = o.Forward(x);
            % backward >> update A matrix of each rule
            o = o.Backward(yhat,y,BP_pars,it);
        end

        %% ------------------------ Rule Remover ---------------------------
        function outNets = RuleRemover(net,InputData,percent,k,opts)
            arguments
                net
                InputData
                percent
                k
                opts.OptimizeParams = 0
                opts.OutputData_required
                opts.DisplayIterations = "iter" % {"iter","none","off","final"}
                opts.PSO_MaxIterations = 50
                opts.PSO_MaxStallIterations = 10
            end

            if opts.OptimizeParams
                if isempty(opts.OutputData_required)
                    error("Need output data for optimize parameters")
                end
            else
                opts.OutputData_required = [];
            end
            [~,~,lambdas] = net.Test(InputData);

            %%%%%%%%%%% Methods Setting %%%%%%%%%%%
            Methods = ["Percentage","MeanMultiStd","MeanMinesStd","new"];
            MethodsPar = [percent, nan, k, nan];
            mustOptimize = [1 0 1 0];
            lb = [0 nan 0 nan];
            ub = [1 nan 5 nan];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % loop on methods
            for i = 1:numel(Methods)
                parameter = MethodsPar(i);
                if opts.OptimizeParams && mustOptimize(i)
                    PSO_options = optimoptions("particleswarm",...
                        "Display", opts.DisplayIterations,...
                        "MaxIterations", opts.PSO_MaxIterations,...
                        "MaxStallIterations", opts.PSO_MaxStallIterations);
                    objFunc = @(parameter) RuleRemoverCore(net,InputData,lambdas,Methods(i),parameter,opts.OutputData_required);
                    parameter = particleswarm(objFunc, 1, lb(i), ub(i), PSO_options);
                    fprintf("Optimized parameter of method(%s) = %f \n",Methods(i),parameter)
                end
                [~,outNets.(Methods(i))] = RuleRemoverCore(net,InputData,lambdas,Methods(i),parameter,opts.OutputData_required);
            end

            % Nested Function
            function [loss,net] = RuleRemoverCore(net,X,lambdas,type,parameter,Y)
                loss = [];
                for l = 1:numel(lambdas)
                    lambdas{l} = mean(lambdas{l},2);

                    if type == "Percentage"
                        removedRules = (lambdas{l} <= prctile(lambdas{l},parameter*100));
                    elseif type == "MeanMultiStd"
                        removedRules = (lambdas{l} < mean(lambdas{l}) * std(lambdas{l}));
                    elseif type == "MeanMinesStd"
                        removedRules = (lambdas{l} < mean(lambdas{l}) - parameter * std(lambdas{l}));
                    elseif type == "new"
                        removedRules = (lambdas{l} < (mean(lambdas{l})+prctile(lambdas{l},50))/(2+std(lambdas{l})));
                    else
                        error("invalid method name")
                    end

                    if prod(removedRules)
                        [~,maxidx] = max(lambdas{l});
                        removedRules(maxidx) = 0;
                    end
                    net.Layer{l}.N = net.Layer{l}.N - sum(removedRules);
                    net.Layer{l}.CS(removedRules) = [];
                    net.Layer{l}.Cc(:,removedRules) = [];
                    net.Layer{l}.CX(removedRules) = [];
                    net.Layer{l}.P(:,removedRules) = [];
                    net.Layer{l}.A(repelem(removedRules,net.Layer{l}.W),:) = [];
                    net.n_rulePerLayer(l) = net.Layer{l}.N;
                end
                if ~isempty(Y)
                    [~,err,~] = net.Test(X,Y);
                    loss = err.LOSS;
                end
            end
        end
    end

    %% PRIVATE FUNCTIONS
    methods (Access=private)
        %%  ------------------------ FORWARD PATH  ------------------------
        function [y,o,BP_pars] = Forward(o,x)
            % x : mini batch data : size(#features,batch_size)

            % Layer
            BP_pars.lambda_4D = cell(1,o.n_Layer);
            BP_pars.AfAx_4D = cell(1,o.n_Layer);
            BP_pars.AfpAx_4D = cell(1,o.n_Layer);
            for l = 1:numel(o.Layer)
                % each data in mini_batch
                MB = size(x,2);
                for k = 1:MB
                    o.Layer{l} = o.RuleFunc(o.Layer{l}, x(:,k), o.dataSeenCounter+k);
                    % o.Layer{l}.X(1:o.Layer{l}.M + 1, k) = [1;x(:,k)];
                end

                % calculate for backward process
                N = o.Layer{l}.N;
                o.n_rulePerLayer(l) = N;
                n = 1:N;
                M = o.Layer{l}.M;
                W = o.Layer{l}.W;

                [lambda_l,tau2_l,Dall] = o.GetLambda(x,o.Layer{l},n);
                % dens_5D = reshape(Dall,1,1,N,MB,numel(o.KernelsName)); %(N,MB,nKernel)
                % BP_pars.tau2_4D{l} = reshape(tau2_l, 1,1,N,1); %(N,1)=>(1,1,N,MB=NoNeed)     %?
                % BP_pars.dens_4D{l} = reshape(sum(Dall,3), 1,1,N,MB); %(N,MB)=>(1,1,N,MB)     %?
                BP_pars.lambda_4D{l} = reshape(lambda_l, 1,1,N,MB); %(N,MB)=>(1,1,N,MB)
                BP_pars.xbar_4D{l} = reshape([ones(1,MB);x], M+1,1,1,MB); %(M+1,MB)=>(M+1,1,1,MB)

                % determin input of next layer
                [x, AfAx, AfpAx] = o.GetOutput(o.Layer{l}, x, lambda_l);
                
                % calculate for backward process
                BP_pars.AfAx_4D{l} = reshape(AfAx, W,1,N,MB); %(W*N,MB)=>(W,1,N,MB)
                BP_pars.AfpAx_4D{l} = reshape(AfpAx, W,1,N,MB); %(W*N,MB)=>(W,1,N,MB)
            end
            y = x;    % last output of layers
        end

        %% ------------------------------------- ADD OR UPDATE RULE  -------
        % used in @forward
        function l = RuleFunc(o,l,xk,dataSeen)
            % l : Layer struct : o.Layer{desire}
            % xk : one data : (#feature,1)
            SEN_xk = o.EucNorm2(xk); % Square Eucdulian Norm of xk

            m = 1:l.M;
            n = 1:l.N;

            % update global information
            l.gMu(m) = l.gMu(m) + (xk - l.gMu(m)) / dataSeen;
            l.gX = l.gX + (SEN_xk - l.gX) / dataSeen;

            if l.N == 0 % Stage(0)
                %%%%% ADD(init) %%%%%
                l = o.InitRule(l,xk,SEN_xk);
                switch o.WeightInitializationType
                    case "xavier"
                        l.A = [l.A ; normrnd(0,2 / (l.M+1 + l.W), l.W, l.M+1)];
                    otherwise
                        l.A = [l.A ; randi([0,1], l.W, l.M+1) / (l.M + 1)];
                end
            else % Stage(1)
                % determine density of xk in all rules in this layer : less distance has more density
                Dl = o.GetDensity(xk,l,n);
                % xk has seen? or new?
                [max_dens,n_star] = max(Dl);
                if max_dens < o.DensityThreshold
                    %%%%% ADD %%%%%
                    l = o.InitRule(l,xk,SEN_xk);
                    switch o.WeightInitializationType
                    case "mean"
                        l.A = [l.A; mean(pagetranspose(reshape(l.A',l.M+1,l.W,[])),3)];
                    case "xavier"
                        l.A = [l.A ; normrnd(0,2 / (l.M+1 + l.W), l.W, l.M+1)];
                    otherwise
                        l.A = [l.A ; randi([0,1], l.W, l.M+1) / (l.M + 1)];
                    end
                else
                    %%%%% UPDATE %%%%%
                    l = o.UpdateRule(l,n_star,xk,SEN_xk);
                end
            end
        end

        %% ------------------------------------- INITIALIZE RULE  ------
        % used in @add_or_update_rule
        function l = InitRule(~,l,xk,SEN_xk)
            % create new cluster
            l.N = l.N + 1;
            % Prototype (Anticident parameters)
            l.P(:, l.N) = xk;
            % Cluster Center
            l.Cc(:, l.N) = xk;
            % Center Square Eucidulian Norm
            l.CX(l.N) = SEN_xk;
            % Number of data in Cluster
            l.CS(l.N) = 1;
        end

        %% ------------------------------------- UPDATE RULE  ----------
        % used in @add_or_update_rule
        function l = UpdateRule(o,l,n_star,xk,SEN_xk)
            m = 1:l.M;

            % add one data in cluster(n_star)
            l.CS(n_star) = l.CS(n_star) + 1;

            % pull cluster(n_star)
            l.Cc(m,n_star) = l.Cc(m, n_star) + (xk - l.Cc(m, n_star)) / l.CS(n_star);
            l.CX(n_star) = l.CX(n_star) + (SEN_xk - l.CX(n_star)) / l.CS(n_star);
            if o.updatePrototype
                l.P(:,n_star) = l.P(:, n_star) + (xk - l.P(:, n_star)) / l.CS(n_star);
            end

            % (new) push other clusters
            % n_other = setdiff(1:l.N,n_star);
            % l.Cc(m,n_other) = l.Cc(m, n_other) - (xk + l.Cc(m, n_other)) ./ l.CS(n_other);
            % l.CX(n_other) = l.CX(n_other) - (SEN_xk + l.CX(n_other)) ./ l.CS(n_other);

            if sum(isnan(l.CX))
                error
            end
        end

        %%  ------------------------- LAYER OUTPUT -------------------------
        % used in @forward
        function [y,yn,AF_prim] = GetOutput(o,l,x,lambda)
            batch_size = size(x,2);
            X = [ones(1,batch_size); x]; % xbar

            % y = sum(lamn-yn,n)
            % lamn_yn = lamn .* yn
            % yn = AF(An*X)
            [yn,AF_prim] = ActFun.(o.ActivationFunction{l.NO})(l.A * X); % (Wl*Nl,Ml+1)*(Ml+1,MB)=(Wl*Nl,MB)
            lamn_yn = repelem(lambda,l.W,1) .* yn;
            y = reshape(sum(reshape(reshape(lamn_yn,l.W,[]), l.W,l.N,[]),2), l.W,[]);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            if max(l.A,[],"all") > 1e1*20
                warning("A big " + max(l.A,[],"all"))
            end
            if sum(isnan(y),"all") || sum(isinf(y),"all")
                error("y "+num2str(sum(y,"all")))
            end
        end

        %% ------------------------- BACKWARD PATH -------------------------
        % update A matrix of each layer
        function o = Backward(o,yh,y,BP_pars,it)
            %   INPUT
            % yh : (nDim,MB)
            % y : (nDim,MB)
            % BP_pars : backpropagation params calculated in forward path
            % it : number of passes iterations

            if o.classification_flag == 2
                y = o.EncodeOneHot(y')';
                yh = softmax(yh);
            end

            % for l = 1:o.n_Layer
            %     dot(o.Layer{l}.A,o.Layer{l}.A)
            % end

            DeDy = yh - y;  % (wl,MB)

            % o.lamReg = o.lamReg - o.LearningRate * sum(o.Layer{end}.A,"all")^2;
            % DeDy = (yh - y) + o.lamReg * sum(o.Layer{end}.A,"all")^2;

            d = cell(1,o.n_Layer);
            d{o.n_Layer} =  reshape(DeDy, o.Layer{end}.W, 1, 1, []); % (W,MB) -> (W,1,1,MB)

            % Rule/N on third dimention
            % MB on forth dimention
            for  l = o.n_Layer : -1 : 1

                N = o.Layer{l}.N;
                n = 1:N;
                M = o.Layer{l}.M;
                W = o.Layer{l}.W;

                xbar_4D = BP_pars.xbar_4D{l}; %(M+1,1,1,MB)
                lambda_4D = BP_pars.lambda_4D{l}; %(1,1,N,MB)
                AfpAx_4D = BP_pars.AfpAx_4D{l}; %(W,1,N,MB)
                AfAx_4D = BP_pars.AfAx_4D{l}; %(W,1,N,MB)

                P_4D = reshape(o.Layer{l}.P(:,n), M,1,N); %(M,N)=>(M,1,N,MB=NoNeed)
                taw2_4D = reshape(o.GetTau2(o.Layer{l},n), 1,1,N); %(N,1)=>(1,1,N,MB=NoNeed)
                %(M,1,N,MB)=2*((M,1,N,1)-(M,1,1,MB))./(1,1,N,1)
                pxt_4D = 2 * (P_4D - xbar_4D(2:end,:,:,:)) ./ taw2_4D;
                %(M,1,N,MB)=(1,1,N,MB).*((M,1,N,MB)-sum((1,1,N,MB).*(M,1,N,MB),3)=(M,1,1,MB))
                DlamDx_4D = lambda_4D .* (pxt_4D - sum(lambda_4D .* pxt_4D,3)); % sum(~,3):sum on N
                %(W,M+1,N,MB)=(1,1,N,MB).*(((W,1,1,MB).*(W,1,N,MB))*(M+1,1,1,MB))
                DeDA = lambda_4D .* PMT((d{l} .* AfpAx_4D),PT(xbar_4D));
                %%%%% sum or mean %%%%%
                DeDA = mean(DeDA,4); %(W,M+1,N,MB)=>(W,M+1,N,1)
                DeDA = reshape(permute(DeDA,[1 3 2]),W*N,M+1); %(W,M+1,N,1)=>(W*N,M+1,1,1)

                %%% Adam Algorithm
                if strcmp(o.SolverName,"Adam")
                    if ~isscalar(o.adapar.m{l})
                        extraMtx = ones(size(DeDA,1)-size(o.adapar.m{l},1),size(DeDA,2));
                        o.adapar.m{l} = [o.adapar.m{l}; extraMtx * o.adapar.ini_m];
                        o.adapar.v{l} = [o.adapar.v{l}; extraMtx * o.adapar.ini_m];
                    end
                    o.adapar.m{l} = o.adapar.b1 * o.adapar.m{l} + (1-o.adapar.b1) * DeDA; % momentum
                    o.adapar.v{l} = o.adapar.b2 * o.adapar.v{l} + (1-o.adapar.b2) * DeDA.^2; % RMSprop
                    % or Algorithm 1
                    mhat = o.adapar.m{l} / (1-o.adapar.b1.^it); % fixing bias
                    vhat = o.adapar.v{l} / (1-o.adapar.b2.^it); % fixing bias
                    DeDA = mhat ./ (sqrt(vhat) + o.adapar.epsilon);
                    % or Algorithm 2
                    % o.LearningRate = sqrt(1-o.adapar.b2.^it) / (1-o.adapar.b1.^it);
                    % DeDA = o.adapar.m{l} ./ (sqrt(o.adapar.v{l}) + o.adapar.epsilon);
                    % o.LearningRate
                end

                % update A
                % o.Layer{l}.A = o.Layer{l}.A - o.LearningRate * DeDA;
                % o.Layer{l}.A = (1-o.LearningRate*o.RegularizationTerm) * o.Layer{l}.A - o.LearningRate * DeDA;
                o.Layer{l}.A = (1-o.LearningRate*o.RegularizationTerm/size(xbar_4D,4)) * o.Layer{l}.A - o.LearningRate * DeDA;

                if sum(isinf(o.Layer{l}.A),"all") || sum(isnan(o.Layer{l}.A),"all")
                    warning
                end

                if l == 1, break, end
                % eq(18) => (W,1,N,MB) : find 'd' of previous layer
                Atild_T_4D = reshape(o.Layer{l}.A(:,2:end)',M,W,N); %(W*N,M)=>(W,M,N,MB=NoNeed)
                %(M,1,N,MB)=sum( (M,1,N,MB)*(1,W,N,MB)*(W,1,1,MB) + (1,1,N,MB).*(W,M,N,1) * ((W,1,1,MB).*(W,1,N,MB)) ,3)
                d{l-1} = sum( PMT(DlamDx_4D,PT(AfAx_4D),d{l}) + PMT(lambda_4D.*Atild_T_4D,(d{l}.*AfpAx_4D)) ,3); % sum(~,3):sum on N
                % d{l-1} = sum( pagemtimes(pagemtimes(DlamDx_4D,AF_T_4D),d{l}) + pagemtimes(lam_4D .* A_tild_3D, d_AFp_4D), 3);
            end

            %%%%%%%%%%% Nested Functions %%%%%%%%%%%
            % pagemtimes
            function out = PMT(A,B,C) 
                if nargin == 2
                    out = pagemtimes(A,B);
                elseif nargin == 3
                    out = pagemtimes(pagemtimes(A,B),C);
                end
            end
            % pagetranspose
            function out = PT(A) 
                out = pagetranspose(A);
            end
        end

        % function DlamDx(o,P_4D,xbar_4D,dens_4D,lambda_4D,tau2_4D,Dall)
        %     x_4D = xbar_4D(2:end,:,:,:);
        %     % (1,1,N,MB)=>(1,1,1,MB)
        %     sumD_i = sum(dens_4D,3);
        %     % (1,1,N,MB)
        %     lambda_4D;
        %     DdDx;
        %     function DdDx
        %         switch type
        %             case "RBF"
        %                 Kp = 2 * (P_4D-x_4D) ./ tau2_4D .* Dall(:,:,:,:,o.KernelsName==type);
        %             case "COS"
        %                 Kp = P_4D ./ sqrt(o.EucNorm2(x_4D)) ./ sqrt(o.EucNorm2(P_4D)) - Dall(:,:,:,:,o.KernelsName==type) .* x_4D ./ o.EucNorm2(x_4D);
        %             case "Linear"
        %                 Kp = o.DotProd(x,l.P(:,n));
        %             case "Polynomial"
        %                 Kp = P_4D;
        %             case "Laplacian"
        %                 Kp = - 1 / sqrt(tau2_4D/2) .* sign(x_4D-P_4D) .* Dall(:,:,:,:,o.KernelsName==type);
        %             case "Chi2"
        %                 Kp = (3*P_4D.^2 - x_4D.^2 - 2*x_4D.*P_4D) ./ (x_4D+P_4D).^2 .* Dall(:,:,:,:,o.KernelsName==type);
        %         end
        %     end
        % end

        %% OTHER FUNCTIONS

        %% ---------------------------- get_tau ----------------------------
        % used in @get_dens
        function tau2 = GetTau2(o,l,n) % (tau_n)^2
            %   INPUT
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % tau2 : for Layer{l} :(#rule,1)

            tau2 = (( abs(l.gX - o.EucNorm2(l.gMu(1:l.M))) + abs(l.CX(n)' - o.EucNorm2(l.Cc(1:l.M,n))) )/2);
            % if prod(taw == 0)
            %     0
            % end
            tau2(tau2 == 0) = eps;
        end

        %% -------------------- SQUARED EUCLIDEAN NORM --------------------
        % used in @get_taw, @update_rule, @init_rule, add_or_update_rule
        function out = EucNorm2(~,x) % ||x||^2
            % IF : x(m,1) => out(1,1) : [SEN(x(m,1))]
            % IF : x(m,n) => out(n,1) : [SEN(x(m,1)); SEN(x(m,2)); ...; SEN(x(m,n))]
            % IF : x(m,n,p) => out(n,p) : [SEN(x(m,1,1), ..., SEN(x(m,1,p));
            %                              ...
            %                              SEN(x(m,n,1)), ..., SEN(x(m,n,p))]
            if numel(size(x))==4 % x:(M,1,(x=1)(p=N),MB)
                [~,~,nRule,nData] = size(x);
                out = zeros(1,1,nRule,nData);
                for k = 1:nData
                    for n = 1:nRule
                        out(1,1,n,k) = dot(x(:,:,n,k),x(:,:,n,k)');
                    end
                end
            else
                [~,nRule,nData] = size(x);
                out = zeros(nRule,nData);
                for k = 1:nData
                    for n = 1:nRule
                        out(n,k) = dot(x(:,n,k),x(:,n,k)');
                    end
                end
            end
        end
        % ----------------- dot prod --------------------
        function out = DotProd(~,x,p) % xp'
            [~,~,nData] = size(x);
            [~,nRule] = size(p);
            out = zeros(nRule,nData);
            for k = 1:nData
                for n = 1:nRule
                    out(n,k) = dot(x(:,1,k),p(:,n));
                end
            end
        end

        %% ------------------------ FIRING STRENGTH ------------------------
        % used in @get_layerOutput, @test
        function [lam,tau,Dall] = GetLambda(o,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % lam : Firing Strength of Cluster(n) with x : (#rule,#data)

            [D,tau,Dall] = o.GetDensity(x,l,n);      % D(#rule,#data)
            lam = D ./ sum(D);          % lam(#rule,#data)
        end

        %% -------------------- LOCAL DENSITY FUNCTION --------------------
        % used in @add_or_update_rule, @get_lam
        function [D,tau2_nl,Dall] = GetDensity(o,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % D : density : (#rule,#data)

            x = reshape(x,size(x,1),1,size(x,2)); %(#feature,1,#data)
            % less distance has more density
            [~,~,nData] = size(x);
            [~,nRule] = size(l.P(:,n));
            nKernel = numel(o.KernelsName);
            p = l.P(:,n);
            tau2_nl = o.GetTau2(l,n); % (N,1)
            euc2_xp = o.EucNorm2(x - p); %(N,MB)

            D = zeros(nRule,nData,nKernel);
            for i = 1:nKernel
                D(:,:,i) = o.KernelsWeight(i) * Kernel(o.KernelsName(i));
            end
            Dall = D;
            % D = sum(D,3);
            D = D(:,:,1); % use RBF only
            D(D == 0) = eps;
            %%%%%%%%%%%%%%%%%%%%%%%%%
            function K = Kernel(type)
                switch type
                    case "RBF"
                        K = exp(- euc2_xp ./ tau2_nl); % exp(-g.euc2(x-p))
                    case "COS"
                        K = o.DotProd(x,p) ./ ( sqrt(o.EucNorm2(x)).*sqrt(o.EucNorm2(p)) ); % x.p'/euc(x).euc(y)
                    case "Linear"
                        K = o.DotProd(x,p); % x'.p
                    case "Polynomial"
                        K = (o.DotProd(x,p) + 1).^2; % (g.x'.p+c)^d
                    case "Laplacian"
                        K = exp(-permute(sum(x-p,1),[2,3,1])./sqrt(tau2_nl/2)); % exp(-g.man(x-p))   % man:manhattan distance
                    case "Chi2"
                        K = exp(- permute(sum((x-p).^2./(x+p)),[2,3,1]) ); % exp( -g.sum((xi-pi)^2/(xi+pi),i) )
                end
            end
        end
        

        %% ------------------------ String Validate ----------------------------
        function o = StringValidate(o)
            o.NormalizeDimension = validatestring(o.NormalizeDimension,["eachFeature","dataFeatures"]);
            for i = 1:numel(o.KernelsName)
                o.KernelsName(i) = validatestring(o.KernelsName(i),["RBF","COS","Linear","Polynomial","Laplacian","Chi2"]);
            end
            o.DataNormalize = validatestring(o.DataNormalize,["none","X","Y","XY"]);
            o.BatchNormType = validatestring(o.BatchNormType,["none","zscore"]);
            o.SolverName = validatestring(o.SolverName,["SGD","Adam","MiniBatchGD","BatchGD","BatchAdam"]);
            o.WeightInitializationType = validatestring(o.WeightInitializationType,["rand","xavier","mean"]);
            for i = 1:numel(o.ActivationFunction)
                o.ActivationFunction(i) = validatestring(o.ActivationFunction(i),["Sigmoid","ReLU","LeakyRelu","Tanh","ELU","Linear","Softmax"]);
            end
            if numel(o.ActivationFunction) == 1
                o.ActivationFunction = repmat(o.ActivationFunction,1,o.n_Layer);
            elseif numel(o.ActivationFunction) == 2
                o.ActivationFunction = [repmat(o.ActivationFunction(1),1,o.n_Layer-1), o.ActivationFunction(2)];
            elseif numel(o.ActivationFunction) ~= o.n_Layer
                error("Incorrect number of AFs")
            end
        end

        %% ------------------------ normalize ----------------------------
        function [X,Y] = NormalizeData(o,X,Y)
            if contains(o.DataNormalize,'X') && exist("X","var")
                if o.NormalizeDimension == "eachFeature"
                    dim = 1;
                elseif o.NormalizeDimension == "dataFeatures"
                    dim = 2;
                end
                X = normalize(X,dim,"range");
            end
            if contains(o.DataNormalize,'Y') && exist("Y","var")
                Y = (Y - o.minY) / (o.maxY - o.minY);
            end
        end
        %% ------------------------ UNnormalize ----------------------------
        function Y = UnNormalizeOutput(o,Y)
            if contains(o.DataNormalize,'Y')
                Y = Y * (o.maxY - o.minY) + o.minY;
            end
        end

        %% ---------------------------- Loss  -----------------------------
        function loss = GetLoss(o,yh,y)
            %   INPUT
            % yh : (nData,nDim)
            % y : (nData,nDim)

            if o.regression_flag % Mean of Squared Error
                loss = mse(yh,y);

            elseif o.classification_flag == 1 % Logestic Regression
                loss = -sum(y .* log(yh + eps) + (1 - y) .* log(1 - yh + eps)) / numel(y);

            elseif o.classification_flag == 2 % Cross Entropy
                % y : one hot vectors
                y = o.EncodeOneHot(y);
                yh = softmax(yh')';
                loss = -mean(sum(y .* log(yh + eps),2));
            end
        end

        %% --------------------------- Metrics ----------------------------
        function metrics = GetMetric(o,yh,y,get_all)
            if nargin < 4, get_all = ""; end
            if o.regression_flag
                if strcmpi(get_all,"all")
                    % Mean Squared Error (MSE)
                    metrics.MSE = mse(yh,y);
                    % Root Mean Squared Error (RMSE)
                    metrics.RMSE = sqrt(metrics.MSE);
                    % NDEI
                    metrics.NDEI = metrics.RMSE / std(y);
                    metrics.NDEI2 = metrics.RMSE / std(yh);
                    % Mean Absolute Error (MAE)
                    metrics.MAE = mae(y,yh);
                    % R-squared (R²)
                    metrics.R2 = 1 - (sum((y - yh).^2) / sum((y - mean(yh)).^2));
                    % Mean Absolute Percentage Error (MAPE)
                    metrics.MAPE = mean(abs((y - yh) ./ y) * 100);
                    % Explained Variance Score
                    metrics.explained_variance = 1 - (var(y - yh) / var(y));
                    % Explained Variance Score
                    metrics.explainedVariance = mean(1 - (sum((y - yh).^2) ./ sum((y - mean(yh)).^2)));
                    % Mean Squared Logarithmic Error (MSLE)
                    metrics.msle = mean( mean((log1p(y) - log1p(yh)).^2));
                    % Mean Bias Deviation (MBD)
                    metrics.mbd = mean( mean(y - yh));
                    % Normalized Root Mean Squared Error (NRMSE)
                    range = max(y) - min(y);
                    metrics.nrmse = mean( metrics.RMSE ./ range);
                    % Mean Percentage Error (MPE)
                    metrics.mpe =  mean(mean((y - yh) ./ y) * 100);
                    % Mean Absolute Percentage Error (MAPE)
                    metrics.mape =  mean(mean(abs((y - yh) ./ y)) * 100);
                else
                    metrics = mse(yh,y);
                end

            elseif o.classification_flag
                cm = confusionmat(y, yh);
                nClass = size(cm,1);
                [TP,FP,FN,TN] = deal(zeros(1,nClass));
                for i = 1:nClass
                    TP(i) = cm(i,i);
                    FP(i) = sum(cm(:, i), 1) - TP(i);
                    FN(i) = sum(cm(i, :), 2) - TP(i);
                    TN(i) = sum(cm(:)) - TP(i) - FP(i) - FN(i);
                end
                if strcmpi(get_all,"all")
                    % Accuracy; [0,1]
                    metrics.ACC = sum(TP) / sum(cm,"all");
                    % Balanced Accuracy
                    metrics.BACC = mean((TP/(TP+FP) + TN/(TN+FP))/2);
                    % Precision; [0,1]
                    metrics.PREC = mean(TP ./ (TP + FP));
                    % Recall; [0,1]
                    metrics.RECALL = mean(TP ./ (TP + FN));
                    % F1 Score; [0,1]
                    metrics.F1SCORE = 2 * (metrics.PREC .* metrics.RECALL) / (metrics.PREC + metrics.RECALL);
                    % Cohen's Kappa
                    agree = sum(TP)/sum(cm(:));
                    chanceAgree = sum((sum(cm,1)/sum(cm(:))) .* (sum(cm,2)/sum(cm(:))));
                    metrics.KAPPA = (agree - chanceAgree) / (1 - chanceAgree);
                    % Matthews Correlation Coefficient (MCC);  [−1,1]
                    s = sum(cm(:));
                    c = sum(TP);
                    p = sum(cm,1);
                    t = sum(cm,2);
                    metrics.MCC = (c.*s - sum(p.*t')) ./ (sqrt((s^2-sum(p.^2)).*(s^2-sum(t.^2)))+eps);
                else
                    metrics = sum(TP) / sum(cm,"all");
                end
            else
                error("Invalid 'ProblemType'.")
            end
        end

        %% ------------------------ Choose Class --------------------------
        function yh = GetClassLabel(o,yh)
            if o.classification_flag == 1
                yh(yh > 0.5) = 1;
                yh(yh <= 0.5) = 0;
            elseif o.classification_flag == 2
                [~,yh] = max(softmax(yh'));
                yh = yh';
            end
        end

        %% ---------------------- CONSTRUCT VARIABLES ---------------------
        % for more speed
        function o = ConstructVars(o)
            max_rule = size(o.Xtrain,1);             % Maximum rule of each layer
            max_layer = o.n_Layer;              % Maximum number of layer

            % Initialize Variables
            o.Layer = cell(1,max_layer);
            for l = 1:max_layer
                o.Layer{l}.NO = l;
                o.Layer{l}.M = o.n_nodes(l);
                o.Layer{l}.W = o.n_nodes(l+1);
                o.Layer{l}.gMu = (zeros(o.Layer{l}.M, 1)); %inf
                o.Layer{l}.gX = zeros;%(o.Layer{l}.M, 1); %inf
                o.Layer{l}.N = 0;
                o.Layer{l}.taw = (zeros(round(max_rule/2), 1)); % /2 ?
                o.Layer{l}.CS = 0;
                o.Layer{l}.Cc = (zeros(o.Layer{l}.M, round(max_rule/2))); % /2 ? %inf
                o.Layer{l}.CX = zeros(1, round(max_rule/2)); % /2 ? %inf
                o.Layer{l}.P = (zeros(o.Layer{l}.M, round(max_rule/2))); % /2 ? %inf
                o.Layer{l}.A = ([]);
                o.Layer{l}.X = (zeros(o.Layer{l}.M + 1, o.MiniBatchSize)); %inf
            end
            o.n_rulePerLayer = zeros(1, o.n_Layer);
            % o.lambda_MB = cell(1,o.n_Layer);
        end

        %% ------------------------------ PLOT -----------------------------
        function GetPlot(o,uptr,downtr,upval,downval,xval,labels)
            subplot(2,2,[1 2])
            plot(1:numel(uptr), uptr, 'DisplayName', "Training "+labels(1),'LineWidth',1.5);
            xlabel('Epoch'); ylabel(labels(1)); title('Training Process'); grid on
            if o.classification_flag
                % ylim([min([uptr,upval]),105])
                ylim([0,105])
                legend('show','Location','southeast')
            else
                % ylim([0,max([uptr,upval])])
                legend('show')
            end
            if o.validation_flag
                hold on;
                % plot(1:numel(uptr), upval, 'DisplayName', 'Validation RMSE','LineStyle','--','Color','k');
                line(xval, upval, 'LineStyle', '--', 'Color', 'k', 'Marker', 'o', ...
                    'MarkerFaceColor', 'k','DisplayName', "Validation "+labels(1), ...
                    'MarkerSize',5);
                hold off;
            end

            subplot(2,2,[3 4])
            plot(1:numel(downtr), downtr, 'DisplayName', "Training "+labels(2),'LineWidth',1.5);
            xlabel('Epoch'); ylabel(labels(2)); legend('show'); grid on
            % ylim([0,max([downtr,downval])])
            if o.validation_flag
                hold on
                % plot(1:numel(uptr), downval, 'DisplayName', 'Validation Loss','LineStyle','--','Color','k');
                line(xval, downval, 'LineStyle', '--', 'Color', 'k', 'Marker', 'o', ...
                    'MarkerFaceColor', 'k','DisplayName', "Validation "+labels(2), ...
                    'MarkerSize',5);
                hold off
            end
            drawnow;
        end

        %% ------------------------- one-hot-vector -----------------------
        function y_onehot = EncodeOneHot(o,y)
            %   INPUT
            % y : (nData,1)
            %   OUTPUT
            % y_onehot : (nData,nClass)

            uniqs = o.uniqLabels;
            if min(uniqs) == 0, uniqs=uniqs+1; end
            nClass = numel(uniqs);
            y_onehot = zeros(size(y,1),nClass);
            for i = 1:nClass
                y_onehot(y==uniqs(i),i) = 1;
            end
        end

        % function y = DecodeOneHot(o,y_onehot)
        %     nClass = size(y_onehot,2);
        %     y = zeros(size(y_onehot,1),1);
        %     classLabels = 1:nClass;
        %     if o.minClassLabel == 0
        %         classLabels = classLabels - 1;
        %     elseif o.minClassLabel > 1
        %         classLabels = classLabels + o.minClassLabel - 1;
        %     end
        %     for i = 1:nClass
        %         y(logical(y_onehot(:,i))) = classLabels(i);
        %     end
        % end

        function [x,y] = checkData(~,x,y)
            nanIdxs = logical(sum([sum(isnan(x),2),sum(isnan(y),2)],2));
            x(nanIdxs,:) = [];
            y(nanIdxs,:) = [];
            if sum(nanIdxs)
                warning("remove "+num2str(sum(nanIdxs))+" NAN row of data")
            end
        end
    end
end