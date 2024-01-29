% MB : Mini Batch
% k : Instance
% l : layer
% n : rule/cluster
% X : xbar : [1;k]
classdef msofnn2
    properties
        n_Layer
        n_rulePerLayer
        n_nodes
        DensityThreshold
        LearningRate
        batchSize
        MaxEpoch
        Layer
        solverName
        ActivationFunction = "Sigmoid"
        BatchNormType
        MSE_report
    end
    properties (Access=private)
        dataSeenCounter
        verbose
        n_data   % Number of training data
        lambda_MB % {layer}(rule,MB)
        AF_AX % AF(A*xbar) : {layer}(Wl,1)
        DlamDx % {layer}(Ml,1)
        AFp_AX % derivative of AF : {layer}()
        adapar % parameters of Adam algorithm
    end
    methods
        function o = msofnn2(Xtr, Ytr, L, pars)
            % Xtr : (#data,#dim)                : Input data
            % Ytr : (#data,#dim)                : Output data
            % L   : (scaler)                    : Number of layers
            % pars : .n_nodes (1,L+1)           : Number of nodes
            %        .MaxEpoch (scaler)         : Maximum epoch
            %        .DensityThreshold (scaler) : Density threshold
            %        .LearningRate (scaler)     : Learning rate
            %        .verbose {0,1}             : show learning steps
            
            % Parameters
            o.n_Layer = L;
            o.n_nodes = pars.n_nodes;
            o.DensityThreshold = pars.DensityThreshold;
            o.LearningRate = pars.LearningRate;
            o.MaxEpoch = pars.MaxEpoch;
            o.batchSize = pars.batchSize;
            clear L pars
            o.n_data = size(Xtr, 1);
            o.dataSeenCounter = 0;
            % Adam Parameters
            o.adapar.ini_m = 0;
            o.adapar.ini_v = 0;
            o.adapar.b1 = 0.9;
            o.adapar.b2 = 0.999;
            o.adapar.epsilon = 1e-8;

            % Prepare Adam Parameters
            o.adapar.m = cell(1,o.n_Layer);
            o.adapar.v = cell(1,o.n_Layer);
            [o.adapar.m{:}] = deal(o.adapar.ini_m);
            [o.adapar.v{:}] = deal(o.adapar.ini_v);

            % initialize variables
            o = o.construct_vars();

            %% ------------------------- Training -------------------------
            iteration = 0;
            % MSE_it = zeros(1,o.MaxEpoch);
            %%%%%%%% epoch %%%%%%%%
            for epoch = 1:o.MaxEpoch
                yhat = zeros(size(Ytr))';

                shuffle_idx = randperm(o.n_data);
                %%%%%%%%%%% iteration %%%%%%%%%%%
                for it = 1 : ceil(o.n_data/o.batchSize)
                    iteration = iteration + 1;
                    MB_idx = shuffle_idx( (it-1)*o.batchSize+1 : min(it*o.batchSize,o.n_data) );

                    x = Xtr(MB_idx,:)';
                    if o.BatchNormType ~= "none"
                        x = normalize(x,2,o.BatchNormType); % dim=2; normalize each feature
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
                    % yhat(:,miniBatch_idx) = yMB;
                end

                %%% epoch - results
                MSE_ep(epoch) = mse(Ytr,yhat');
                m=MSE_ep(epoch)
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
            if size(Xtest,1) ~= net.n_nodes(1)
                Xtest = Xtest';
            end
            
            x = Xtest;
            for l = 1:net.n_Layer
                rules = 1:net.n_rulePerLayer(l);
                lambda =  net.get_lam(x, net.Layer{l}, rules);
                x = net.get_layerOutput(net.Layer{l}, x, lambda);
            end
            yhat = x;
            
            if exist("Ytest","var")
                if size(Ytest,1) ~= net.n_nodes(end)
                    Ytest = Ytest';
                end
                err.MSE = mse(yhat,Ytest);
                err.RMSE = sqrt(err.MSE);
                STD = std(yhat);%?
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
                % each data in mini_batch
                for k = 1:size(x,2)
                    o.Layer{l} = o.add_or_update_rule(o.Layer{l}, x(:,k), o.dataSeenCounter+k);
                    o.Layer{l}.X(1:o.Layer{l}.M + 1, k) = [1;x(:,k)];
                end

                n = 1:o.Layer{l}.N;
                o.lambda_MB{l} =  o.get_lam(x,o.Layer{l},n); % (rule,MB)
                taw_nl = o.get_taw(o.Layer{l},n); % (rule,1)
                pxt = 2 * ( reshape(o.Layer{l}.P(:,n),[],1,n(end)) - x ) ./ reshape(taw_nl,1,1,n(end)); % (Ml,MB,rule)
                o.DlamDx{l} = reshape(o.lambda_MB{l}',1,[],n(end)) .* (pxt - sum(pxt,3)); % (Ml,MB,rule)

                % determin input of next layer
                [x, AfAx, AfpAx] = o.get_layerOutput(o.Layer{l}, x, o.lambda_MB{l});
                o.n_rulePerLayer(l) = o.Layer{l}.N;

                o.AF_AX{l} = pagetranspose(reshape(AfAx', size(AfAx,2), o.Layer{l}.W, [])); % (Wl*Nl,MB) -> (Wl,MB,Nl)
                o.AFp_AX{l} = pagetranspose(reshape(AfpAx', size(AfpAx,2), o.Layer{l}.W, [])); % (Wl*Nl,MB) -> (Wl,MB,Nl)
                
            end
            y = x;    % last output of layers
        end

        % ---------------------- ADD OR UPDATE RULE  ----------------------
        % used in @forward
        function l = add_or_update_rule(o,l,xk,dataSeen)
            % l : Layer struct : o.Layer{desire}
            % xk : one data : (#feature,1)
            SEN_xk = norm(xk).^2; % Square Eucdulian Norm of xk

            m = 1:l.M;
            n = 1:l.N;
            if l.N == 0 % Stage(0)
                %%%%% ADD(init) %%%%%
                l.gMu(m) = xk;
                l.gX = SEN_xk;
                l = o.init_rule(l,xk,SEN_xk);
            else % Stage(1)
                % update global information
                l.gMu(m) = l.gMu(m) + (xk - l.gMu(m)) / dataSeen;
                l.gX = l.gX + (SEN_xk - l.gX) / dataSeen;

                % determine density of xk in all rules in this layer : less distance has more density
                Dl = o.get_dens(xk,l,n);

                % xk has seen? or new?
                [max_dens,n_star] = max(Dl);
                if max_dens < o.DensityThreshold
                    %%%%% ADD %%%%%
                    l = o.init_rule(l,xk,SEN_xk);
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
            l.A = [l.A ; randi([0,1], l.W, l.M+1) / (l.M + 1)];
            % Prototype (Anticident parameters)
            l.P(1:l.M, l.N) = xk;
            % Cluster Center
            l.Cc(1:l.M, l.N) = xk;
            % Center Square Eucidulian Norm
            l.CX(l.N) = SEN_xk;
            % Number of data in Cluster
            l.CS(l.N) = 1;
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
        function [y,AfAx,AfpAx] = get_layerOutput(o,l,x,lambda)
            % y_l = sum(lam_nl * y_nl)
            % y_nl = AF(A_nl * xbar_l)

            batch_size = size(x,2);
            % xk : (#feature,#instance)
            X = [ones(1,batch_size); x];
            
            % tic
            %%% SOLUTION 1
            [AfAx,AfpAx] = ActFun.(o.ActivationFunction)(l.A * X); % (Wl*Nl,Ml+1)*(Ml+1,MB)=(Wl*Nl,MB)
            ynl = reshape(AfAx,l.W*l.N,1,[]); % (Wl*Nl,1,MB)
            lam_l = zeros(l.W, l.W*l.N, batch_size);
            for k = 1:batch_size
                for n = 1:l.N
                    lam_l(:,(n-1)*l.W+1:n*l.W,k) = diag(ones(1,l.W)*lambda(n,k));
                end
            end
            y = reshape(pagemtimes(lam_l,ynl),[],batch_size);
            % toc
            % y

            % clear y
            % tic
            % %%% SOLUTION 2
            % ynl = o.AF(l.A * X); %(Wl*Nl,1)
            % ynl = reshape(ynl,l.W,l.N);
            % newlam = zeros(l.W*l.N, batch_size);
            % for n = 1:l.N
            %     newlam((n-1)*l.W+1:n*l.W) = repmat(lambda(n,:),l.W,1);
            % end
            % lam_ynl = newlam .* ynl;
            % for n = 1:l.W
            %     y(n,:) = sum(lam_ynl(n:l.W:l.W*l.N));
            % end
            % toc
            % y
            % 
            % clear y
            % tic
            % %%% SOLUTION 3
            % ynl = o.AF(l.A * X); %(Wl*Nl,1)
            % lam_l{k} = deal(zeros(l.W, l.W*l.N));
            % for k = 1:batch_size
            %     for n = 1:l.N
            %         lam_l{k}(:,(n-1)*l.W+1:n*l.W) = diag(ones(1,l.W)*lambda(n,k));
            %     end
            %     y(:,k) = lam_l{k} * ynl(:,k);
            % end
            % toc
            % y
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
                if strcmp(o.solverName,"adam")
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
            
            taw = sqrt(( abs(l.gX - norm(l.gMu(1:l.M))^2) + abs(l.CX(n)' - o.squEucNorm(l.Cc(1:l.M,n))) )/2);
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
                    out(n,k) = norm(x(:,n,k)).^2;
                end
            end
            % toc


            % tic
            % sq = sqrt(pagemtimes(pagetranspose(x),x));
            % out = zeros(size(x,2),size(x,3));
            % for i = 1:size(x,3)
            %     out(:,i) = diag(sq(:,:,i)).^2;
            % end
            % toc
            % prod(prod(out1 == out))
            % out1 == out
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
            D = exp(- o.squEucNorm(x - l.P(1:l.M,n)) ./ o.get_taw(l,n).^2);
        end

        % ---------------------- ACTIVATION FUNCTION ----------------------
        % function out = AF(o,in)
        %     switch o.ActivationFunction
        %         case "Sigmoid"
        %             out = 1 ./ (1 + exp(-in));
        %         case "Tanh"
        %             out = 
        %         case "Relu"
        %             out = max(in,0);
        %         case "LeakyRelu"
        %             alpha = 0.01;
        %             out = max(in * alpha, in);
        %     end
        % end

        % -------------------------- AF_derivative --------------------------
        function out = AF_derivative(o,in)
            switch o.ActivationFunction
                case "Sigmoid"
                    out = in .* (1 - in);
                case "Relu"
                    in(in >= 0) = 1;
                    in(in < 0) = 0;
                    out = in;
                case "LeakyRelu"
                    alpha = 0.01;
                    in(in > 0) = 1;
                    in(in < 0) = 0;
                    out = in;
                    out = max(in * alpha, in);
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
                % o.Layer{l}.NO = l;
                o.Layer{l}.M = o.n_nodes(l);
                o.Layer{l}.W = o.n_nodes(l+1);
                o.Layer{l}.gMu = inf(o.Layer{l}.M, 1);
                o.Layer{l}.gX = inf;
                o.Layer{l}.N = 0;
                o.Layer{l}.taw = zeros(max_rule/2, 1); % /2 ?
                o.Layer{l}.CS = 0;
                o.Layer{l}.Cc = inf(o.Layer{l}.M, max_rule/2); % /2 ?
                o.Layer{l}.CX = inf(1, max_rule/2); % /2 ?
                o.Layer{l}.P = inf(o.Layer{l}.M, max_rule/2); % /2 ?
                o.Layer{l}.A = [];
                o.Layer{l}.X = inf(o.Layer{l}.M + 1, o.batchSize);
            end
            o.n_rulePerLayer = zeros(1, o.n_Layer);
            o.lambda_MB = cell(1,o.n_Layer);
        end
    end
end