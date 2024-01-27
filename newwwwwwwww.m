classdef msofnn
    properties
        Layer
        dataSeenCounter
        delta
    end
    methods
        function obj = msofnn(X)
            batch_size = 3;
            batch_idx = reshape([randperm(K),K+1:K+batch_size-rem(K,batch_size)],batch_size,[]);
            l.A = []
            l.N = 0
            obj.dataSeenCounter = 0
            l.taw = zeros(:,1)
            % main
            iteration = 0;
            % epoch
            for epoch = 1:EPOCH
                % iteration
                for miniBatch_idx = batch_idx
                    miniBatch_idx(miniBatch_idx>K) = [];
                    if isempty(miniBatch_idx), break, end
                    iteration = iteration + 1;
                    MB = Xtr(miniBatch_idx,:)';
                    SEN = obj.squEucNorm(MB);
                    % forward >> Create Rules and estimaye final output
                    [yMB,Layer] = obj.forward(MB(:,miniBatch_idx));
                    obj.dataSeenCounter = obj.dataSeenCounter + numel(miniBatch_idx);
                    % backward >> update A matrix of each rule
                    obj = obj.backward(yMB,Ytr(miniBatch_idx,:)');
                end
            end
        end

        %% TEST
        function test(o,Layer,x)
            x = x'; %(#feature,#data)
            get_dens_lam()
            for k = 1:size(x,2)

            end
            X = [ones(1,size(x,2)); x];

        end
    end

    methods (Access=private)
        %% FORWARD PATH
        function [y,obj] = forward(obj,x)
            % x : mini batch data : size(#features,batch_size)

            % Layer
            for l = 1:numel(obj.Layer)
                % each data in mini_batch
                for k = 1:size(x,2)
                    obj.Layer{l} = obj.add_or_update_rule(obj.Layer{l}, x(:,k), o.dataSeenCounter+k);
                    obj.Layer{l}.X(1:obj.Layer{l}.M,k) = [1;x(:,k)];
                end
                % determin input of next layer
                x = obj.get_layerOutput(obj.Layer{l},x);
            end
            y = x;    % last output of layers
        end

        %%% ADD OR UPDATE RULE
        % used in @forward
        function l = add_or_update_rule(obj,l,xk,dataSeen)
            % l : Layer struct : obj.Layer{desire}
            % xk : one data : (#feature,1)

            m = 1:l.M;
            n = 1:l.N;
            if l.N == 0 % Stage(0)
                %%%%% ADD(init) %%%%%
                l.gMu(m) = xk;
                l.gX = obj.squEucNorm(xk);
                l = obj.init_rule(l,xk);
            else % Stage(1)
                % update global information
                l.gMu(m) = l.gMu(m) + (xk - l.gMu(m)) / dataSeen;
                l.gX = l.gX + (obj.squEucNorm(xk) - l.gX) / dataSeen;

                % determine density of xk in all rules in this layer : less distance has more density
                Dl = obj.get_dens(xk,l,n);

                % xk has seen? or new?
                [max_dens,n_star] = max(Dl);
                if max_dens < o.delta
                    %%%%% ADD %%%%%
                    l = obj.init_rule(l,xk);
                else
                    %%%%% UPDATE %%%%%
                    l = obj.update_rule(l,n_star,xk);
                end
            end
        end

        %%% INITIALIZE RULE
        % used in @add_or_update_rule
        function l = init_rule(~,l,xk)
            % create new cluster
            l.N = l.N + 1;
            % Conceqent parameters
            l.A = [l.A ; randi([0,1], l.W, l.M+1) / (l.M + 1)];
            % Prototype (Anticident parameters)
            l.P(1:l.M, l.N) = xk;
            % Cluster Center
            l.Cc(1:l.M, l.N) = xk;
            % Center Square Eucidulian Norm
            l.CX(l.N) = o.squEucNorm(xk);
            % Number of data in Cluster
            l.CS(l.N) = 1;
            %
            l.taw(l.N) = obj.update_taw(l,l.N);
        end

        %%% UPDATE RULE
        % used in @add_or_update_rule
        function l = update_rule(obj,l,n_star,xk)
            m = 1:l.M;
            SEN_xk = obj.squEucNorm(xk);

            % add one data in cluster(n_star)
            l.CS(n_star) = l.CS(n_star) + 1;

            % pull cluster(n_star)
            l.Cc(m,n_star) = l.Cc(m, n_star) + (xk - l.Cc(m, n_star)) / l.CS(n_star);
            l.CX(n_star) = l.CX(n_star) + (SEN_xk - l.CX(n_star)) / l.CS(n_star);

            % (new) push other clusters
            n_other = setdiff(1:l.N,n_star);
            l.Cc(m,n_other) = l.Cc(m, n_other) - (xk - l.Cc(m, n_other)) ./ l.CS(n_other)';
            l.CX(n_other) = l.CX(n_other) - (SEN_xk - l.CX(n_other)) ./ l.CS(n_other);

            % update taw
            l.taw(:,1) = obj.update_taw(l,1:l.N);
        end

        %% LAYER OUTPUT
        % used in @forward
        function get_layerOutput(obj,l,x)
            % y_l = sum(lam_nl * y_nl)
            % y_nl = AF(A_nl * xbar_l)

            batch_size = size(x,2);
            % xk : (#feature,#instance)
            X = [ones(1,batch_size); x];
            lambda =  obj.get_lam(obj,x,l,1:l.N); % size(#rule,batch_size)

            %%% SOLUTION 1
            ynl = AF(l.A * X); %(Wl*Nl,1)
            ynl = reshape(ynl,l.W,1,[]);
            lam_l = zeros(l.W, l.W*l.N, batch_size);
            for k = 1:batch_size
                for n = 1:l.N
                    lam_l(:,(n-1)*l.W+1:n*l.W,k) = diag(ones(1,l.W)*lambda(n,l));
                end
            end
            y = pagemtimes(lam_l,ynl);
            y

            clear y
            %%% SOLUTION 2
            ynl = AF(l.A * X); %(Wl*Nl,1)
            ynl = reshape(ynl,l.W,l.N);
            newlam = zeros(l.W*l.N, batch_size);
            for n = 1:l.N
                newlam((n-1)*l.W+1:n*l.W) = repmat(lambda(n,:),l.W,1);
            end
            lam_ynl = newlam .* ynl;
            for n = 1:l.W
                y(n,:) = sum(lam_ynl(n:l.W:l.W*l.N));
            end
            y

            clear y
            %%% SOLUTION 3
            ynl = AF(l.A * X); %(Wl*Nl,1)
            lam_l{k} = deal(zeros(l.W, l.W*l.N));
            for k = 1:batch_size
                for n = 1:l.N
                    lam_l{k}(:,(n-1)*l.W+1:n*l.W) = diag(ones(1,l.W)*lambda(n,k));
                end
                y(:,k) = lam_l{k} * ynl(:,k);
            end
            y
        end

        %% BACKWARD PATH
        % update A matrix of each layer
        function obj = backward(obj,y_hat,y_target)
        end

        %% OTHER FUNCTIONS

        %%% update_taw
        % used in @init_rule, @update_rule
        function taw = update_taw(obj,l,n)
            %   INPUT
            % l : Layer struct : obj.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % taw : for Layer{l} :(#rule,1)

            taw = sqrt(( abs(l.gX - obj.squEucNorm(l.gMu(1:l.M))) + abs(l.CX(n) - obj.squEucNorm(l.Cc(1:l.M,n))) )/2);
            taw(taw == 0) = eps;
        end

        %%% SQUARED EUCLIDEAN NORM
        function out = squEucNorm(~,x)
            sq = sqrt(pagemtimes(pagetranspose(x),x));
            out = zeros(size(x,2),size(x,3));
            for i = 1:size(x,3)
                out(:,i) = diag(sq(:,:,i)).^2;
            end
        end

        %%% FIRING STRENGTH OF X IN RULES
        % used in @get_layerOutput, @test
        function lam = get_lam(obj,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : obj.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % lam : Firing Strength of Cluster(n) with x : (#rule,#data)

            D = obj.get_dens(x,l,n);
            lam = D ./ sum(D);
        end

        %%% LOCAL DENSITY FUNCTION
        % used in @add_or_update_rule, @get_lam
        function D = get_dens(obj,x,l,n)
            %   INPUT
            % x : data : (#features,#data)
            % l : Layer struct : obj.Layer{desire}
            % n : scaler or vector : (1,#rule)
            %   OUTPUT
            % D : density : (#rule,#data)

            x = reshape(x,size(x,1),1,size(x,2));
            % less distance has more density
            D = exp(- obj.squEucNorm(x - l.P(1:l.M,n)) ./ l.taw(n).^2);
        end
    end
end