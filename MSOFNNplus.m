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
        Layer
        solverName
        ActivationFunction
        WeightInitializationType
        DataNormalize
        BatchNormType
        MSE_report
    end
    properties (Access=private)
        plot
        Xtrain
        Ytrain
        dataSeenCounter
        verbose
        n_data   % Number of training data
        lambda_MB % {layer}(rule,MB)
        AF_AX % AF(A*xbar) : {layer}(Wl,1)
        DlamDx % {layer}(Ml,1)
        AFp_AX % derivative of AF : {layer}()
        adapar % parameters of Adam algorithm
        minYtr
        maxYtr
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
            opts.DataNormalize = validatestring(opts.DataNormalize,["none","X","XY"]);
            opts.BatchNormType = validatestring(opts.BatchNormType,["none","zscore"]);
            opts.SolverName = validatestring(opts.SolverName,["SGD","Adam","MiniBatchGD"]);
            opts.WeightInitializationType = validatestring(opts.WeightInitializationType,["none","xavier"]);
            for i = 1:numel(opts.ActivationFunction)
                opts.ActivationFunction(i) = validatestring(opts.ActivationFunction(i),["Sigmoid","ReLU","LeakyRelu","Tanh","ELU","Linear"]);
            end

            if numel(opts.ActivationFunction) == 1
                opts.ActivationFunction = repmat(opts.ActivationFunction,1,n_Layer);
            elseif numel(opts.ActivationFunction) == 2
                opts.ActivationFunction = [repmat(opts.ActivationFunction(1),1,n_Layer-1), opts.ActivationFunction(2)];
            elseif numel(opts.ActivationFunction) ~= n_Layer
                error("Incorrect number of AFs")
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

            % Parameters
            o.n_Layer = n_Layer;
            o.n_nodes = [M,W(end)];
            o.LearningRate = opts.LearningRate; 
            o.MaxEpoch = opts.MaxEpoch;
            o.DensityThreshold = opts.DensityThreshold;
            o.verbose = opts.verbose;
            o.plot = opts.plot;
            o.ActivationFunction = opts.ActivationFunction;
            o.WeightInitializationType = opts.WeightInitializationType;
            o.BatchNormType = opts.BatchNormType;
            o.DataNormalize = opts.DataNormalize;
            o.solverName = opts.SolverName;
            o.MiniBatchSize = opts.MiniBatchSize;
            o.n_data = size(Xtr, 1);
            

            if strcmp(o.solverName,"Adam")
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

            elseif strcmp(o.solverName,"SGD") && o.MiniBatchSize > 1
                warning("In SGD Algorithm batch_size should be 1; I fixed this for you")
                o.MiniBatchSize = 1;
            end

            % initialize variables
            o = o.construct_vars();

            % save data
            o.Xtrain = Xtr;
            o.Ytrain = Ytr;
            o.minYtr = min(Ytr);
            o.maxYtr = max(Ytr);
        end

        %% ------------------------- Training -------------------------
        function trained_net = train(o,opts)
            arguments
                o
                opts.validationPercent = 0
            end
            if opts.validationPercent
                idx = randperm(size(o.Xtrain,1));
                n_val = round(size(o.Xtrain,1) * opts.validationPercent);
                Xval = o.Xtrain(idx(1:n_val),:);
                Yval = o.Ytrain(idx(1:n_val),:);
                o.Xtrain = o.Xtrain(idx(n_val+1:end),:);
                o.Ytrain = o.Ytrain(idx(n_val+1:end),:);
                MSE_val_best = inf;
            end
            [Xtr,Ytr] = o.dataNormalize(o.Xtrain,o.Ytrain);
            
            iteration = 0;
            MSE_ep = zeros(1,o.MaxEpoch);
            %%%%%%%% epoch %%%%%%%%
            for epoch = 1:o.MaxEpoch
                o.dataSeenCounter = 0;
                yhat = zeros(size(Ytr))';

                shuffle_idx = randperm(o.n_data);
                %%%%%%%%%%% iteration %%%%%%%%%%%
                for it = 1 : ceil(o.n_data/o.MiniBatchSize)
                    iteration = iteration + 1;
                    MB_idx = shuffle_idx( (it-1)*o.MiniBatchSize+1 : min(it*o.MiniBatchSize,o.n_data) );

                    x = Xtr(MB_idx,:)';
                    if ~strcmp(o.BatchNormType,"none")
                        x = normalize(x,1,o.BatchNormType); % dim=2; normalize each feature
                    end
                    y = Ytr(MB_idx,:)';

                    % Forward >> Backward
                    [o,yhat_MB] = o.main(x,y,iteration);

                    % save pars
                    yhat(:,MB_idx) = yhat_MB;
                    o.dataSeenCounter = o.dataSeenCounter + numel(MB_idx);

                    %%% iteration - results
                    % MSE_it(iteration) = mse(yhat_MB,y);
                    % plot(1:iteration,MSE_it)
                    % drawnow
                    % mse(yhat_MB,y)
                end

                %%% epoch - results
                mse(Ytr,yhat')
                yhat = o.outputUnNormalize(yhat);
                mse(o.Ytrain,yhat')
                MSE_ep(epoch) = mse(o.Ytrain,yhat');
                % disp([epoch, MSE_ep(epoch)])

                if opts.validationPercent
                    [~,errVal] = o.test(Xval,Yval);
                    if (epoch>1) && (errVal.mse < MSE_val_best)
                        netBest = o;
                        MSE_val_best = errVal.mse;
                    end
                end
                if (epoch>5) && prod(MSE_ep(epoch-5:end) - MSE_ep(epoch)), break, end
            end

            if opts.validationPercent
                trained_net.last = o;
                trained_net.best = netBest;
            else
                trained_net = o;
            end

            [bestMSE,bestIdx] = min(MSE_ep);
            meanMSE = mean(MSE_ep);
            o.MSE_report.Mean = meanMSE;
            o.MSE_report.Best = bestMSE;
            o.MSE_report.Last = MSE_ep(end);

            MSE = ["Last";"Best";"Mean"];
            Value = [MSE_ep(end); bestMSE; meanMSE];
            Epoch = [epoch; bestIdx; nan];
            table(MSE,Value,Epoch)
            % o.MSE_report = sprintf("[Mean:%.3f, Best:%.3f]",mean(MSE_ep),min(MSE_ep));
        end

        %% ----------------------------- TEST -----------------------------
        function [yhat, err] = test(net,Xtest,Ytest)
            [Xtest,Ytest] = net.dataNormalize(Xtest,Ytest);
            Xtest = Xtest';

            % if ~strcmp(net.BatchNormType, "none")
            %     Xtest = normalize(Xtest,2,net.BatchNormType); % dim=2; normalize each feature
            % end

            x = Xtest;
            for l = 1:net.n_Layer
                rules = 1:net.n_rulePerLayer(l);
                lambda =  net.get_lam(x, net.Layer{l}, rules);
                x = net.get_layerOutput(net.Layer{l}, x, lambda);
            end
            yhat = net.outputUnNormalize(x');

            if exist("Ytest","var")
                err.MSE = mse(yhat,Ytest);
                err.RMSE = sqrt(err.MSE);
                STD = std(Ytest);%?
                err.NDEI = err.RMSE / STD;
            end
        end

        %%  ----------------------------- main -----------------------------
        function [o,yhat] = main(o,x,y,it)
            % forward >> Create Rules and estimaye final output
            [yhat,o] = o.forward(x);
            % backward >> update A matrix of each rule
            o = o.backward(yhat,y,it);
        end
    end

    methods (Access=private)
        %%  ------------------------ FORWARD PATH  ------------------------
        function [y,o] = forward(o,x)
            % x : mini batch data : size(#features,batch_size)

            % Layer
            o.lambda_MB = cell(1,o.n_Layer);
            o.AF_AX = cell(1,o.n_Layer);
            o.AFp_AX = cell(1,o.n_Layer);
            for l = 1:numel(o.Layer)
                % x = (x - o.minX) ./ (o.maxX - o.minX);
                % y = (y - minY) ./ (maxY - minY);
                % x = normalize(x);

                % each data in mini_batch
                for k = 1:size(x,2)
                    o.Layer{l} = o.add_or_update_rule(o.Layer{l}, x(:,k), o.dataSeenCounter+k);
                    o.Layer{l}.X(1:o.Layer{l}.M + 1, k) = [1;x(:,k)];
                end
                o.n_rulePerLayer(l) = o.Layer{l}.N;

                % for backward process
                n = 1:o.Layer{l}.N;
                o.lambda_MB{l} =  o.get_lam(x,o.Layer{l},n); % (N,MB)
                taw_nl = o.get_taw(o.Layer{l},n); % (N,1)
                pxt = 2 * ( reshape(o.Layer{l}.P(:,n),[],1,n(end)) - x ) ./ reshape(taw_nl,1,1,n(end)); % (M,MB,N)
                lam_1MBn = reshape(o.lambda_MB{l}',1,[],n(end)); % (1,MB,N)
                o.DlamDx{l} = lam_1MBn .* (pxt - sum(lam_1MBn.*pxt,3)); % (M,MB,N)

                % determin input of next layer
                [x, AfAx, AfpAx] = o.get_layerOutput(o.Layer{l}, x, o.lambda_MB{l});

                % for backward process
                o.AF_AX{l} = pagetranspose(reshape(AfAx', size(AfAx,2), o.Layer{l}.W, [])); % (W*N,MB) -> (W,MB,N)
                o.AFp_AX{l} = pagetranspose(reshape(AfpAx', size(AfpAx,2), o.Layer{l}.W, [])); % (W*N,MB) -> (W,MB,N)

            end
            y = x;    % last output of layers
        end

        % ---------------------- ADD OR UPDATE RULE  ----------------------
        % used in @forward
        function l = add_or_update_rule(o,l,xk,dataSeen)
            % l : Layer struct : o.Layer{desire}
            % xk : one data : (#feature,1)
            SEN_xk = o.squEucNorm(xk); % Square Eucdulian Norm of xk

            m = 1:l.M;
            n = 1:l.N;

            % update global information
            l.gMu(m) = l.gMu(m) + (xk - l.gMu(m)) / dataSeen;
            l.gX = l.gX + (SEN_xk - l.gX) / dataSeen;
            % l.GloDelta = abs(o.squEucNorm(l.gMu(m)) - l.gX);

            if l.N == 0 % Stage(0)
                %%%%% ADD(init) %%%%%
                l = o.init_rule(l,xk,SEN_xk);
                switch o.WeightInitializationType
                    case "xavier"
                        l.A = [l.A ; normrnd(0,2 / (l.M+1 + l.W), l.W, l.M+1)];
                    otherwise
                        l.A = [l.A ; randi([0,1], l.W, l.M+1) / (l.M + 1)];
                end
            else % Stage(1)
                % determine density of xk in all rules in this layer : less distance has more density
                Dl = o.get_dens(xk,l,n);

                % xk has seen? or new?
                [max_dens,n_star] = max(Dl);
                if max_dens < o.DensityThreshold
                    %%%%% ADD %%%%%
                    l = o.init_rule(l,xk,SEN_xk);
                    l.A = [l.A; mean(pagetranspose(reshape(l.A',l.M+1,l.W,[])),3)];
                else
                    %%%%% UPDATE %%%%%
                    l = o.update_rule(l,n_star,xk,SEN_xk);
                end
            end
        end

        % ------------------------ INITIALIZE RULE  ------------------------
        % used in @add_or_update_rule
        function l = init_rule(~,l,xk,SEN_xk)
            % create new cluster
            l.N = l.N + 1;
            % Conceqent parameters
            % switch o.WeightInitializationType
            %     case "xavier"
            %     l.A = [l.A ; normrnd(0,2 / (l.M+1 + l.W), l.W, l.M+1)];
            %     otherwise
            %     l.A = [l.A ; randi([0,1], l.W, l.M+1) / (l.M + 1)];
            % end
            % Prototype (Anticident parameters)
            l.P(1:l.M, l.N) = xk;

            % Cluster Center
            l.Cc(:, l.N) = xk;
            % Center Square Eucidulian Norm
            l.CX(l.N) = SEN_xk;
            % Number of data in Cluster
            l.CS(l.N) = 1;
            % Local information
            % l.LocDelta(l.N) = abs(o.squEucNorm(l.Cc(:, l.N)) - l.CX(l.N));
        end

        % -------------------------- UPDATE RULE  --------------------------
        % used in @add_or_update_rule
        function l = update_rule(~,l,n_star,xk,SEN_xk)
            m = 1:l.M;

            % add one data in cluster(n_star)
            l.CS(n_star) = l.CS(n_star) + 1;

            % pull cluster(n_star)
            l.Cc(m,n_star) = l.Cc(m, n_star) + (xk - l.Cc(m, n_star)) / l.CS(n_star);
            l.CX(n_star) = l.CX(n_star) + (SEN_xk - l.CX(n_star)) / l.CS(n_star);

            % % (new) push other clusters
            % n_other = setdiff(1:l.N,n_star);
            % l.Cc(m,n_other) = l.Cc(m, n_other) - (xk - l.Cc(m, n_other)) ./ l.CS(n_other);
            % l.CX(n_other) = l.CX(n_other) - (SEN_xk - l.CX(n_other)) ./ l.CS(n_other);
        end

        %%  ------------------------- LAYER OUTPUT -------------------------
        % used in @forward
        function [y,yn,AF_prim] = get_layerOutput(o,l,x,lambda)
            % y_l = sum(lam_nl * y_nl)
            % y_nl = AF(A_nl * xbar_l)

            batch_size = size(x,2);
            % xk : (#feature,#instance)
            X = [ones(1,batch_size); x];

            % tic
            %%% SOLUTION 1
            % y = sum(lamn-yn,n)
            % lamn_yn = lamn .* yn
            % yn = AF(An*X)
            % if max(l.A,[],"all") > 1e1*20
                % max(l.A,[],"all")
                % warning("A big")
            % end
            [yn,AF_prim] = ActFun.(o.ActivationFunction{l.NO})(l.A * X); % (Wl*Nl,Ml+1)*(Ml+1,MB)=(Wl*Nl,MB)
            lamn_yn = repelem(lambda,l.W,1) .* yn;
            y = reshape(sum(reshape(reshape(lamn_yn,l.W,[]), l.W,l.N,[]),2), l.W,[]);
            % ynl = reshape(AfAx,l.W*l.N,1,[]); % (Wl*Nl,1,MB)
            % lam_l = zeros(l.W, l.W*l.N, batch_size);
            % for k = 1:batch_size
            %     for n = 1:l.N
            %         lam_l(:,(n-1)*l.W+1:n*l.W,k) = diag(ones(1,l.W)*lambda(n,k));
            %     end
            % end
            % y = reshape(pagemtimes(lam_l,ynl),[],batch_size);

            if sum(isnan(y),"all") || sum(isinf(y),"all")
                warning("y inf nan")
            end

        end

        %% ------------------------- BACKWARD PATH -------------------------
        % update A matrix of each layer
        function o = backward(o,y_hat,y_target,it)
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
                if strcmp(o.solverName,"Adam")
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
                    % lr = o.LearningRate
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

        % ---------------------------- get_taw ----------------------------
        % used in @get_dens
        function taw = get_taw(o,l,n)
            %   INPUT
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % taw : for Layer{l} :(#rule,1)

            taw = (( abs(l.gX - o.squEucNorm(l.gMu(1:l.M))) + abs(l.CX(n)' - o.squEucNorm(l.Cc(1:l.M,n))) )/2);
            % if prod(taw == 0)
            %     0
            % end
            taw(taw == 0) = eps;
        end

        % -------------------- SQUARED EUCLIDEAN NORM --------------------
        % used in @get_taw, @update_rule, @init_rule, add_or_update_rule
        function out = squEucNorm(~,x)
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

        % ------------------------ FIRING STRENGTH ------------------------
        % used in @get_layerOutput, @test
        function lam = get_lam(o,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % lam : Firing Strength of Cluster(n) with x : (#rule,#data)

            D = o.get_dens(x,l,n);      % D(#rule,#data)
            lam = D ./ sum(D);          % lam(#rule,#data)
        end

        % -------------------- LOCAL DENSITY FUNCTION --------------------
        % used in @add_or_update_rule, @get_lam
        function D = get_dens(o,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : o.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % D : density : (#rule,#data)

            x = reshape(x,size(x,1),1,size(x,2));
            % less distance has more density
            D = exp(- o.squEucNorm(x - l.P(1:l.M,n)) ./ o.get_taw(l,n));
        end

        % ------------------------ normalize ----------------------------
        function [X,Y] = dataNormalize(o,X,Y)
            if ~strcmp(o.DataNormalize,"none")
                X = normalize(X,1,"range");
            end
            if strcmp(o.DataNormalize,"XY")
                Y = (Y - o.minYtr) / (o.maxYtr-o.minYtr);
            end
        end
        % ------------------------ UNnormalize ----------------------------
        function Y = outputUnNormalize(o,Y)
            if strcmp(o.DataNormalize,"XY")
                Y = Y * (o.maxYtr-o.minYtr) + o.minYtr;
            end
        end

        % ---------------------- CONSTRUCT VARIABLES ----------------------
        % for more speed
        function o = construct_vars(o)
            max_rule = o.n_data;             % Maximum rule of each layer
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
                o.Layer{l}.taw = (zeros(max_rule/2, 1)); % /2 ?
                o.Layer{l}.CS = 0;
                o.Layer{l}.Cc = (zeros(o.Layer{l}.M, max_rule/2)); % /2 ? %inf
                o.Layer{l}.CX = zeros(1, max_rule/2); % /2 ? %inf
                o.Layer{l}.P = (zeros(o.Layer{l}.M, max_rule/2)); % /2 ? %inf
                o.Layer{l}.A = ([]);
                o.Layer{l}.X = (zeros(o.Layer{l}.M + 1, o.MiniBatchSize)); %inf
            end
            o.n_rulePerLayer = zeros(1, o.n_Layer);
            o.lambda_MB = cell(1,o.n_Layer);
        end
    end
end