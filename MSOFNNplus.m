% MB : Mini Batch
% k : Instance
% l : layer
% n : rule/cluster
% X : xbar : [1;k]
classdef MSOFNNplus
    properties
        n_Layer
        n_rulePerLayer
        n_nodes
        DensityThreshold
        LearningRate
        MiniBatchSize
        MaxEpoch
        TrainedEpoch
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
    end
    properties (Access=private)
        Xtrain
        Ytrain
        dataSeenCounter
        lambda_MB % {layer}(rule,MB)
        AF_AX % AF(A*xbar) : {layer}(Wl,1)
        DlamDx % {layer}(Ml,1)
        AFp_AX % derivative of AF : {layer}()
        adapar % parameters of Adam algorithm
        uniqOutputs
        classification = 0
        regression = 0
        minY
        maxY
    end
    methods
        function o = MSOFNNplus(Xtr, Ytr, n_Layer, opts)
            arguments
                Xtr % (#data,#dim)                : Input data
                Ytr % (#data,#dim)                : Output data
                n_Layer (1,1) {mustBeInteger,mustBePositive}
                opts.n_hiddenNodes (1,:) {mustBeInteger,mustBePositive,mustBeVector} = ones(1,n_Layer-1)*3*size(Ytr,2)
                opts.LearningRate (1,1) {mustBePositive} = 0.001
                opts.MaxEpoch (1,1) {mustBeInteger,mustBePositive} = 500
                opts.DensityThreshold (1,1) {mustBePositive} = exp(-3)
                opts.verbose (1,1) {logical} = 0
                opts.plot (1,1) {logical} = 0
                opts.ActivationFunction = "Sigmoid"
                opts.BatchNormType {mustBeTextScalar} = "none"
                opts.SolverName {mustBeTextScalar} = "SGD"
                opts.WeightInitializationType {mustBeTextScalar} = "none"
                opts.MiniBatchSize {mustBeInteger,mustBePositive} = 1
                opts.DataNormalize = "none"
                opts.adampar_epsilon = 1e-8
                opts.adampar_beta1 = 0.9
                opts.adampar_beta2 = 0.999
                opts.adampar_m0 = 0
                opts.adampar_v0 = 0
            end

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

            % Validate String Variables
            o = o.StringValidate();

            % Problem Type
            o.uniqOutputs = unique(Ytr);
            if all(isinteger(Ytr)) % Classification
                o.classification = 1;
                if all(ismember(o.uniqOutputs, [0, 1]))
                    disp('Problem Type: Binary Classification');
                    o.ProblemType = "Binary-Classification";
                elseif numel(o.uniqOutputs) > 2
                    disp('Problem Type: Multiclass Classification');
                    o.ProblemType = "Multiclass-Classification";
                end
            else % Regression
                o.regression = 1;
                if size(Ytr,2) == 1
                    disp('Problem Type: MISO Regression');
                    o.ProblemType = "MISO-Regression";
                else
                    disp('Problem Type: MIMO Regression');
                    o.ProblemType = "MIMO-Regression";
                end
            end

            % w check conditions
            W = [opts.n_hiddenNodes,size(Ytr,2)];
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
            if strcmp(o.SolverName,"Adam")
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
                opts.Xval
                opts.Yval
                % or
                opts.validationSplitPercent = 0
                % optional
                opts.valPerEpochFrequency = 1 % test val every 1 epochs
                opts.ApplyRuleRemover {logical} = 0
            end
            % Error check
            if ~opts.validationSplitPercent && opts.ApplyRuleRemover
                warning("Validation data not exist, Rule-Remover will work with training data, which it may havent good performance.")
            end
            if opts.validationSplitPercent && (exist(opts.Xval,"var") || exist(opts.Yval,"var"))
                if (exist(opts.Xval,"var") && exist(opts.Yval,"var"))
                    warning("Validation data entered, so set 'valPerEpochFrequency' to zero.");
                else
                    error("Enter both of Xval and Yval")
                end
            end
            o = o.StringValidate();
            if o.Plot, figure; end
            valData = [];
            if opts.validationSplitPercent || exist(opts.Xval,"var") % Validation
                if exist(opts.Xval,"var")
                    Xval = opts.Xval;
                    Yval = opts.Yval;
                    n_val = size(Xval,1);
                else
                    idx = randperm(size(o.Xtrain,1));
                    n_val = round(size(o.Xtrain,1) * opts.validationSplitPercent);
                    Xval = o.Xtrain(idx(1:n_val),:);
                    Yval = o.Ytrain(idx(1:n_val),:);
                end
                o.Xtrain = o.Xtrain(idx(n_val+1:end),:);
                o.Ytrain = o.Ytrain(idx(n_val+1:end),:);
                bestValLoss = inf;
                valEpochs = [1,opts.valPerEpochFrequency:opts.valPerEpochFrequency:o.MaxEpoch];
                METRIC_val = zeros(1,numel(valEpochs));
                LOSS_val = zeros(1,numel(valEpochs));
                valCount = 0;
            end
            % normalize data if needed
            [Xtr,Ytr] = o.NormalizeData(o.Xtrain,o.Ytrain);

            it = 0; % iteration total
            METRIC_tr = zeros(1,o.MaxEpoch);
            LOSS_tr = zeros(1,o.MaxEpoch);
            
            %%%%%%%% epoch %%%%%%%%
            for epoch = 1:o.MaxEpoch

                % (o.Layer{:}).CS = deal(0);
                
                o.dataSeenCounter = 0;
                yhat = zeros(size(Ytr));
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
                yhat = o.UnNormalizeOutput(yhat);
                LOSS_tr(epoch) = o.GetLoss(yhat,o.Ytrain);
                % if classification
                if endsWith(lower(o.ProblemType),'classification')
                    yhat = o.GetClassLabel(yhat);
                end
                METRIC_tr(epoch) = o.GetMetric(yhat,o.Ytrain); % MSE/ACC for regression/classification

                % verbose
                if o.Verbose
                    fprintf("[Epoch:%d] [MSE:%.4f] [RMSE:%.4f] [Loss:%.4f] \n", epoch, METRIC_tr(epoch), sqrt(METRIC_tr(epoch)), LOSS_tr(epoch))
                elseif ~rem(epoch,10)
                    fprintf("[Epoch:%d]\n", epoch)
                end

                % validation
                if opts.validationSplitPercent && ismember(epoch,valEpochs)
                    valCount = valCount + 1;
                    [~,errVal] = o.Test(Xval,Yval);
                    METRIC_val(valCount) = errVal.MSE;
                    LOSS_val(valCount) = errVal.Loss;
                    if (valCount>1) && (LOSS_val(valCount) < bestValLoss)
                        netBest = o;
                        bestValLoss = LOSS_val(valCount);
                    end
                end

                % Plot
                if (o.Plot) && (epoch > 1)
                    o.GetPlot(opts, METRIC_tr(1:epoch), LOSS_tr(1:epoch), METRIC_val(1:valCount), LOSS_val(1:valCount), valEpochs(1:valCount))
                end

                o.TrainedEpoch = epoch;
                % Eliminate Condition
                if (epoch>5) && prod(METRIC_tr(epoch-5:end) - METRIC_tr(epoch)), break, end

                for l = 1:numel(o.Layer)
                    o.Layer{l}.CS = zeros(size(o.Layer{l}.CS));
                end
            end %%%%%%%%%%% END EPOCH %%%%%%%%%%%

            Apply Rule Remover (if needed)
            if opts.validationSplitPercent
                o = o.RuleRemover(Xval,0.5,0.5);
                netBest = netBest.RuleRemover(Xval,0.5,0.5);
            else
                o = o.RuleRemover(o.Xtrain,0.5,0.5);
            end

            % save network
            if opts.validationSplitPercent
                trained_net.last = o;
                trained_net.best = netBest;
            else
                trained_net = o;
            end

            % return results
            [bestMSE,bestIdx] = min(METRIC_tr);
            meanMSE = mean(METRIC_tr);
            o.MSE_report.Mean = meanMSE;
            o.MSE_report.Best = bestMSE;
            o.MSE_report.Last = METRIC_tr(end);

            METRIC_tr = ["Last";"Best";"Mean"];
            Value = [METRIC_tr(end); bestMSE; meanMSE];
            Epoch = [epoch; bestIdx; nan];
            table(METRIC_tr,Value,Epoch)
            % o.MSE_report = sprintf("[Mean:%.3f, Best:%.3f]",mean(MSE_ep),min(MSE_ep));
            valData.x = Xval;
            valData.y = Yval;
        end

        %% ----------------------------- TEST -----------------------------
        function [yhat, err, lambda] = Test(net,Xtest,Ytest,opts)
            arguments
                net
                Xtest
                Ytest = []
                opts.Plot {logical} = 0
            end
            err = [];
            lambda = cell(1,net.n_Layer);

            Yexist = 1;
            if isempty(Ytest), Yexist = 0; end
            net = net.StringValidate();

            % normalize data if needed
            Xtest = net.NormalizeData(Xtest);
            Xtest = Xtest';

            % if ~strcmp(net.BatchNormType, "none")
            %     Xtest = normalize(Xtest,2,net.BatchNormType); % dim=2; normalize each feature
            % end

            x = Xtest;
            % Forward
            for l = 1:net.n_Layer
                lambda{l} =  net.GetLambda(x, net.Layer{l}, 1:net.n_rulePerLayer(l));
                x = net.GetOutput(net.Layer{l}, x, lambda{l});
            end
            % un normalize if needed
            yhat = net.UnNormalizeOutput(x');

            % determin loss (required Ytest)
            if Yexist, Loss = net.GetLoss(yhat,Ytest); end

            % prepare output for classification problem
            if net.classification,  yhat = net.GetClassLabel(yhat); end

            if Yexist
                % determin Metric (required Ytest)
                [metric1,metric2] = deal([]);
                [metric1,metric2] = net.GetMetric(yhat,Ytest);
                err.Loss = Loss;
                if net.regression
                    err.MSE = metric1;
                    err.RMSE = sqrt(metric1);
                    STD = std(Ytest);
                    err.NDEI = err.RMSE / STD;
                    err.NDEI2 = err.RMSE / (sqrt(net.dataSeenCounter)*(STD));
                    if opts.Plot
                        figure
                        plot(Ytest), hold on, plot(yhat)
                    end
                elseif net.classification
                    err.ACC = metric1;
                end
            end
        end

        %%  ----------------------------- main -----------------------------
        function [o,yhat] = Main(o,x,y,it)
            % forward >> Create Rules and estimaye final output
            [yhat,o] = o.Forward(x);
            % backward >> update A matrix of each rule
            o = o.Backward(yhat,y,it);
        end

        %% ------------------------ Rule Remover ---------------------------
        function outNets = RuleRemover(net,InputData,percent,k,newpercent,opts)
            arguments
                net
                InputData
                percent
                k
                newpercent
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
            MethodsPar = [percent, nan, k, newpercent];
            mustOptimize = [1 0 1 1];
            lb = [0 nan 0 0];
            ub = [1 nan 5 1];
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
                        removedRules = (lambdas{l} < (mean(lambdas{l})+prctile(lambdas{l},parameter*100))/(2+std(lambdas{l})));
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
                    loss = err.Loss;
                end
            end
        end
    end

    %% PRIVATE FUNCTIONS
    methods (Access=private)
        %%  ------------------------ FORWARD PATH  ------------------------
        function [y,o] = Forward(o,x)
            % x : mini batch data : size(#features,batch_size)

            % Layer
            o.lambda_MB = cell(1,o.n_Layer);
            o.AF_AX = cell(1,o.n_Layer);
            o.AFp_AX = cell(1,o.n_Layer);
            for l = 1:numel(o.Layer)
                % each data in mini_batch
                for k = 1:size(x,2)
                    o.Layer{l} = o.RuleFunc(o.Layer{l}, x(:,k), o.dataSeenCounter+k);
                    o.Layer{l}.X(1:o.Layer{l}.M + 1, k) = [1;x(:,k)];
                end
                o.n_rulePerLayer(l) = o.Layer{l}.N;

                % calculate for backward process
                n = 1:o.Layer{l}.N;
                o.lambda_MB{l} =  o.GetLambda(x,o.Layer{l},n); % (N,MB)
                taw_nl = o.GetTaw(o.Layer{l},n); % (N,1)
                pxt = 2 * ( reshape(o.Layer{l}.P(:,n),[],1,n(end)) - x ) ./ reshape(taw_nl,1,1,n(end)); % (M,MB,N)
                lam_1MBn = reshape(o.lambda_MB{l}',1,[],n(end)); % (1,MB,N)
                o.DlamDx{l} = lam_1MBn .* (pxt - sum(lam_1MBn.*pxt,3)); % (M,MB,N)

                % determin input of next layer
                [x, AfAx, AfpAx] = o.GetOutput(o.Layer{l}, x, o.lambda_MB{l});

                % calculate for backward process
                o.AF_AX{l} = pagetranspose(reshape(AfAx', size(AfAx,2), o.Layer{l}.W, [])); % (W*N,MB) -> (W,MB,N)
                o.AFp_AX{l} = pagetranspose(reshape(AfpAx', size(AfpAx,2), o.Layer{l}.W, [])); % (W*N,MB) -> (W,MB,N)

            end
            y = x;    % last output of layers
        end

        % ---------------------- ADD OR UPDATE RULE  ----------------------
        % used in @forward
        function l = RuleFunc(o,l,xk,dataSeen)
            % l : Layer struct : o.Layer{desire}
            % xk : one data : (#feature,1)
            SEN_xk = o.SquEucNorm(xk); % Square Eucdulian Norm of xk

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
                    l.A = [l.A; mean(pagetranspose(reshape(l.A',l.M+1,l.W,[])),3)];
                else
                    %%%%% UPDATE %%%%%
                    l = o.UpdateRule(l,n_star,xk,SEN_xk);
                end
            end
        end

        % ------------------------ INITIALIZE RULE  ------------------------
        % used in @add_or_update_rule
        function l = InitRule(~,l,xk,SEN_xk)
            % create new cluster
            l.N = l.N + 1;
            % Prototype (Anticident parameters)
            l.P(1:l.M, l.N) = xk;
            % Cluster Center
            l.Cc(:, l.N) = xk;
            % Center Square Eucidulian Norm
            l.CX(l.N) = SEN_xk;
            % Number of data in Cluster
            l.CS(l.N) = 1;
        end

        % -------------------------- UPDATE RULE  --------------------------
        % used in @add_or_update_rule
        function l = UpdateRule(~,l,n_star,xk,SEN_xk)
            m = 1:l.M;

            % add one data in cluster(n_star)
            l.CS(n_star) = l.CS(n_star) + 1;

            % pull cluster(n_star)
            l.Cc(m,n_star) = l.Cc(m, n_star) + (xk - l.Cc(m, n_star)) / l.CS(n_star);
            l.CX(n_star) = l.CX(n_star) + (SEN_xk - l.CX(n_star)) / l.CS(n_star);

            % (new) push other clusters
            % n_other = setdiff(1:l.N,n_star);
            % l.Cc(m,n_other) = l.Cc(m, n_other) - (xk + l.Cc(m, n_other)) ./ l.CS(n_other);
            % l.CX(n_other) = l.CX(n_other) - (SEN_xk + l.CX(n_other)) ./ l.CS(n_other);
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
                warning("y inf nan")
            end
        end

        %% ------------------------- BACKWARD PATH -------------------------
        % update A matrix of each layer
        function o = Backward(o,y_hat,y_target,it)
            DeDy = y_hat - y_target;  % (wl,MB)
            % ek = DeDy' * DeDy / 2;

            d = cell(1,o.n_Layer);
            d{o.n_Layer} =  reshape(DeDy, o.Layer{end}.W, 1, 1, []); % (W,MB) -> (W,1,1,MB)

            % Rule/N on third dimention
            % MB on forth dimention
            for  l = o.n_Layer : -1 : 1

                % (M,MB,N) -> (M,1,N,MB)
                DlamDx_4D = reshape(permute(o.DlamDx{l},[1 3 2]),o.Layer{l}.M,1,o.Layer{l}.N,[]);
                % (W,MB,N) -> (W,1,N,MB) -> (1,W,N,MB)
                AF_T_4D = pagetranspose(reshape(permute(o.AF_AX{l},[1 3 2]),o.Layer{l}.W,1,o.Layer{l}.N,[]));
                % (N,MB) -> (1,1,N,MB)
                lam_4D = reshape(o.lambda_MB{l},1,1,o.Layer{l}.N,[]);
                % (W,MB,N) -> (W,1,N,MB)
                AFp_4D = reshape(permute(o.AFp_AX{l},[1 3 2]),o.Layer{l}.W,1,o.Layer{l}.N,[]);
                % A : (W*N,M+1) -> (W*N,M) -> (W,M,N)
                A_tild_3D = reshape(o.Layer{l}.A(:,2:end)', o.Layer{l}.M, o.Layer{l}.W, []);

                % save param
                d_AFp_4D = d{l} .* AFp_4D;

                % eq(17) => (W,M+1,N,MB)
                DeDA = pagemtimes( lam_4D .* d_AFp_4D, pagetranspose(reshape(o.Layer{l}.X(:,1:size(y_hat,2)),o.Layer{l}.M+1,1,1,[])) );
                % mean => (W,M+1,N)
                % DeDA = mean(DeDA,4); % mean of error of mini batch
                DeDA = sum(DeDA,4); % sum of error of mini batch
                % reshape => (W*N,M+1)
                DeDA = reshape(pagetranspose(permute(DeDA,[3 1 2])),[],size(DeDA,2));

                %%% Adam Algorithm
                if strcmp(o.SolverName,"Adam")
                    if ~isscalar(o.adapar.m{l})
                        extraMtx = ones(size(DeDA,1)-size(o.adapar.m{l},1),size(DeDA,2));
                        o.adapar.m{l} = [o.adapar.m{l}; extraMtx * o.adapar.ini_m];
                        o.adapar.v{l} = [o.adapar.v{l}; extraMtx * o.adapar.ini_m];
                    end
                    o.adapar.m{l} = o.adapar.b1 * o.adapar.m{l} + (1-o.adapar.b1) * DeDA;
                    o.adapar.v{l} = o.adapar.b2 * o.adapar.v{l} + (1-o.adapar.b2) * DeDA.^2;

                    % or Algorithm 1
                    % mhat = o.adapar.m{l} / (1-o.adapar.b1.^it);
                    % vhat = o.adapar.v{l} / (1-o.adapar.b2.^it);
                    % DeDA = mhat ./ (sqrt(vhat) + o.adapar.epsilon);

                    % or Algorithm 2
                    o.LearningRate = sqrt(1-o.adapar.b2.^it) / (1-o.adapar.b1.^it);
                    DeDA = o.adapar.m{l} ./ (sqrt(o.adapar.v{l}) + o.adapar.epsilon);
                end

                % update A
                o.Layer{l}.A = o.Layer{l}.A - o.LearningRate * DeDA;

                if sum(isinf(o.Layer{l}.A),"all") || sum(isnan(o.Layer{l}.A),"all")
                    warning
                end

                if l == 1, break, end
                % eq(18) => (W,1,N,MB) : find 'd' of previous layer
                d{l-1} = sum( pagemtimes(pagemtimes(DlamDx_4D,AF_T_4D),d{l}) + pagemtimes(lam_4D .* A_tild_3D, d_AFp_4D), 3);
            end

        end

        %% OTHER FUNCTIONS

        %% ---------------------------- get_taw ----------------------------
        % used in @get_dens
        function taw = GetTaw(o,l,n)
            %   INPUT
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % taw : for Layer{l} :(#rule,1)

            taw = (( abs(l.gX - o.SquEucNorm(l.gMu(1:l.M))) + abs(l.CX(n)' - o.SquEucNorm(l.Cc(1:l.M,n))) )/2);
            % if prod(taw == 0)
            %     0
            % end
            taw(taw == 0) = eps;
        end

        %% -------------------- SQUARED EUCLIDEAN NORM --------------------
        % used in @get_taw, @update_rule, @init_rule, add_or_update_rule
        function out = SquEucNorm(~,x)
            % IF : x(m,1) => out(1,1) : [SEN(x(m,1))]
            % IF : x(m,n) => out(n,1) : [SEN(x(m,1)); SEN(x(m,2)); ...; SEN(x(m,n))]
            % IF : x(m,n,p) => out(n,p) : [SEN(x(m,1,1), ..., SEN(x(m,1,p));
            %                              ...
            %                              SEN(x(m,n,1)), ..., SEN(x(m,n,p))]

            % tic
            [~,nRule,nData] = size(x);
            out = zeros(nRule,nData);
            for k = 1:nData
                for n = 1:nRule
                    out(n,k) = dot(x(:,n,k),x(:,n,k)');
                end
            end
            % toc


            % % tic
            % sq = (pagemtimes(pagetranspose(x),x));
            % out = zeros(size(x,2),size(x,3));
            % for i = 1:size(x,3)
            %     out(:,i) = diag(sq(:,:,i)).^2;
            % end
            % % toc
        end

        %% ------------------------ FIRING STRENGTH ------------------------
        % used in @get_layerOutput, @test
        function lam = GetLambda(o,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % lam : Firing Strength of Cluster(n) with x : (#rule,#data)

            D = o.GetDensity(x,l,n);      % D(#rule,#data)
            lam = D ./ sum(D);          % lam(#rule,#data)
        end

        %% -------------------- LOCAL DENSITY FUNCTION --------------------
        % used in @add_or_update_rule, @get_lam
        function D = GetDensity(o,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % D : density : (#rule,#data)

            x = reshape(x,size(x,1),1,size(x,2)); %(#feature,1,#data)
            % less distance has more density
            D = exp(- o.SquEucNorm(x - l.P(1:l.M,n)) ./ o.GetTaw(l,n));

            %%%%%%%%%%%%%%%%%
            % % % Calculate the covariance matrix
            % Sigma = cov(x);
            % % % Extract the variances, which are the diagonal elements of the covariance matrix
            % variances = diag(Sigma);
            % D = zeros(l.N,size(x,2));
            % for k = 1:size(x,2)
            %     for n = 1:l.N
            %         D(n,k) = (x(:,k) - l.P(:,n))' .* variances(k) * (x(:,k) - l.P(:,n));
            %     end
            % end
        end

        %% ------------------------ String Validate ----------------------------
        function o = StringValidate(o)
            o.DataNormalize = validatestring(o.DataNormalize,["none","X","Y","XY"]);
            o.BatchNormType = validatestring(o.BatchNormType,["none","zscore"]);
            o.SolverName = validatestring(o.SolverName,["SGD","Adam","MiniBatchGD"]);
            o.WeightInitializationType = validatestring(o.WeightInitializationType,["none","xavier"]);
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
                X = normalize(X,1,"range");
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

        %% ----------------------- Loss Function ---------------------------
        function loss = GetLoss(o,yh,y)
            epsilon = 1e-15;  % Small constant to avoid numerical instability
            switch lower(o.ProblemType)
                case 'miso-regression'
                    try
                    loss = sum((yh - y).^2,1) / 2;
                    catch
                        0
                    end
                case 'mimo-regression'
                    loss = sum((yh - y).^2,"all") / size(y,2);
                case 'binary-classification'
                    yh = max(epsilon, min(1 - epsilon, yh));  % Clip to avoid log(0)
                    loss = -mean(y .* log(yh) + (1 - y) .* log(1 - yh));
                case 'multiclass-classification'
                    yh = max(epsilon, yh);  % Clip to avoid log(0)
                    loss = -mean(sum(y .* log(yh), 2));
                otherwise
                    error("Invalid 'ProblemType'.")
            end
        end
        %% ---------------------- Metrics Function -------------------------
        function [metric1,metric2] = GetMetric(o,yh,y)
            metric2 = [];
            if contains(lower(o.ProblemType),'regression')
                metric1 = mse(yh,y); % MSE
            elseif contains(lower(o.ProblemType),'classification')
                cm = confusionmat(yh,y);
                metric1 = sum(diag(cm)) / sum(cm(:)); % ACCURACY
            else
                error("Invalid 'ProblemType'.")
            end
        end

        %% ------------------------ Choose Class ---------------------------
        function yh = GetClassLabel(o,yh)
            if strcmpi(o.ProblemType,'binary-classification')
                yh(yh > 0.5) = 1;
                yh(yh <= 0.5) = 0;
            elseif strcmpi(o.ProblemType,'multiclass-classification')
                %%% MAX
                % [~,yh] = max(yh);
                %%% SOFTMAX
                [~,yh] = max(softmax(yh));
            else
                error("Invalid 'ProblemType'.")
            end
        end

        %% ---------------------- CONSTRUCT VARIABLES ----------------------
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
            o.lambda_MB = cell(1,o.n_Layer);
        end

        %% ------------------------------ PLOT -----------------------------
        function GetPlot(o,opts,uptr,downtr,upval,downval,xval)
            if endsWith(lower(o.ProblemType),'regression') % regression
                subplot(2,2,[1 2])
                plot(1:numel(uptr), uptr, 'DisplayName', 'Training RMSE','LineWidth',1.5);
                ylabel('RMSE'); title('Training Process'); legend('show'); grid on
                if (opts.validationSplitPercent)
                    hold on;
                    % plot(1:numel(uptr), upval, 'DisplayName', 'Validation RMSE','LineStyle','--','Color','k');
                    line(xval, upval, 'LineStyle', '--', 'Color', 'k', 'Marker', 'o', 'MarkerFaceColor', 'k','DisplayName', 'Validation RMSE');
                    hold off;
                end

                subplot(2,2,[3 4])
                plot(1:numel(uptr), downtr, 'DisplayName', 'Training Loss','LineWidth',1.5);
                xlabel('Epoch'); ylabel('Loss'); legend('show'); grid on
                if (opts.validationSplitPercent)
                    hold on
                    % plot(1:numel(uptr), downval, 'DisplayName', 'Validation Loss','LineStyle','--','Color','k');
                    line(xval, downval, 'LineStyle', '--', 'Color', 'k', 'Marker', 'o', 'MarkerFaceColor', 'k','DisplayName', 'Validation Loss');
                    hold off
                end
                drawnow;
            else % classification
                subplot(2,2,[1 2])
                plot(1:numel(uptr), uptr, 'DisplayName', 'Training Acc','LineWidth',1.5);
                xlabel('Epoch'); ylabel('Acuracy'); title('Training Process'); legend('show','Location','best'); grid on
                if (opts.validationSplitPercent)
                    hold on;
                    plot(1:numel(uptr), upval, 'DisplayName', 'Validation Acc','LineStyle','--','Color','k');
                    hold off;
                end

                subplot(2,2,[3 4])
                plot(1:numel(upval), downtr, 'DisplayName', 'Training Loss','LineWidth',1.5);
                xlabel('Epoch'); ylabel('Loss'); legend('show'); grid on
                if (opts.validationSplitPercent)
                    hold on;
                    plot(1:numel(uptr), downval, 'DisplayName', 'Validation Loss','LineStyle','--','Color','k');
                    hold off;
                end
                drawnow;
            end

        end

    end
end