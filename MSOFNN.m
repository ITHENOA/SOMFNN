classdef MSOFNN
    properties
        W       % Number of layer outputs
        M       % Number of layer inputs
        delta   % Distance threshold
        gMu     % Global mean of all the input samples
        N       % Number of each leyer's rule
        gX      % Global mean of the squared Euclidean norms of all the input sample
        K       % Number of instances
        Cc      % mean(center)s of clusters
        CX      % Square Euclidean norms of each instance of each rule of each layer
        CS      % Number of data in each cluster of each layer
        P       % Anticident(Prototype) Pars
        A       % Conciquent Pars
        xb      % xbar [1;xkl]
        lambda  % Firing strength of "xl" to "Rnl"
        D       % Local density of "xkl" at each cluster
        L       % Number of layers
        d       % backprobagation parameter
        alpha   % Learning rate
        % yhat    % estimated output
        EPOCH   % number of epochs
        ActivationFunction = "sigmoid"
    end
    properties (Access=private)
        Xtrain
        Ytrain
    end
    
    methods
        function o = MSOFNN(Xtrain,Ytrain,L,w,delta,alpha)
            arguments
                Xtrain %(#instance, #dim/feature)
                Ytrain %(#instance, #output) : if (#output>1) then we have MIMO system
                L (1,1) {mustBePositive} % number of layers
                w (1,:) {mustBePositive} = ones(1,L-1)*3*size(Ytrain,2); % (1,L-1) or (1,1) : number of output of each layer exept last one
                delta (1,1) {mustBePositive} = exp(-3) % thereshold
                alpha (1,1) {mustBePositive} = 0.01 % Learning Rate
            end

            % construct W and M
            if numel(w) == L - 1
                o.W = [w, size(Ytrain,2)]; %(1,l)
            elseif numel(w) == L 
                if w(end) ~= size(Ytrain,2)
                    w(end) = size(Ytrain,2);
                end
                o.W = w;
            else
                error("incorrect number of w's elements")
            end
            o.L = L;
            o.M = [size(Xtrain,2), o.W(1:end-1)]; %(1,l)

            % w check conditions
            if sum(o.M < o.W)
                [~,l] = find(o.M < o.W);
                warning(['Number of outputs of layer (%s) may cause overfitting \n' ...
                    'Fix this: (W(%d)=%d <= %d) \nHint: W(l) <= M(l)'], ...
                    num2str(l), l(1), o.W(l(1)), o.M(l(1)));
            end
            if sum(o.W < o.W(end))
                [~,l] = find(o.W < o.W(end));
                warning(['Number of outputs of layer (%s) may cause lossing too much informations \n' ...
                    'Fix this: (W(%d)=%d >= %d) \nHint: W(l) >= output dimension'], ...
                    num2str(l), l(1), o.W(l(1)), o.W(end));
            end

            % o.EPOCH = EPOCH;
            o.alpha = alpha;
            o.K = size(Xtrain, 1);     % Number of instances
            max_input = max(o.M);       % Maximum input of layers
            max_rule = o.K;             % Maximum rule of each layer
            max_layer = L;              % Maximum number of layer

            % Initialize Variables
            o.N = zeros(1, max_layer);                              % N(1,l)
            o.delta = delta;                                        % delta(1,1)
            o.gMu = inf(max_input, max_layer);                      % gMu(dim,l)
            o.gX = inf(1, max_layer);                               % gX(1,l)
            o.Cc = inf(max_input, max_rule, max_layer);             % Cc(dim,n,l)
            o.CX = inf(max_input, max_layer);                       % CX(n,l)
            o.CS = inf(max_rule, max_layer);                        % CS(n,l)
            o.P = inf(max_input, max_rule, max_layer);              % P(Ml,n,l)
            o.A = inf(max_input, max_input+1, max_rule, max_layer); % A(Wl,Ml+1,n,l)
            o.xb = inf(max_input+1, max_layer);                     % xbar(Ml+1,1)  for k
            o.lambda = inf(max_rule, max_layer);                    % lambda(n,l)
            o.D = inf(max_rule, max_layer);                         % D(n,l)
            o.d = zeros(max_input, max_layer);                      % d(wl,l)   for k
            
            o.xb(1,:) = ones(1,max_layer);
            o.Xtrain = Xtrain;
            o.Ytrain = Ytrain;
        end

        %% TRAIN
        function o = train(o,EPOCH,verbose)
            arguments
                o
                EPOCH (1,1) {mustBeInteger,mustBePositive} = 100 % Epoch
                verbose = 1 % show plots
            end

            % MSE = inf(EPOCH,1);
            for epoch = 1:EPOCH
                % epoch
                [o,yhat] = o.main(o.Xtrain, o.Ytrain, epoch);
                % [o,MSE(epoch),~] = o.main(o.Xtrain, o.Ytrain, epoch);
                % if (epoch > 1) && (abs(MSE(epoch) - MSE(epoch-1)) < 1e-6), break, end
                % if verbose
                %     plot(1:epoch,MSE(1:epoch))
                %     title("epoch="+epoch+" mse="+MSE(epoch))
                %     xlabel('Epoch')
                %     ylabel('MSE')
                %     drawnow
                % end
                mse(o.Ytrain,yhat)
            end
            % fprintf("epoch=%d, mse=%d",epoch,MSE(epoch))
        end

        %% TEST
        function [yhat, err] = test(o, X_test, y_test)
            xl = X_test';
            for l = 1:o.L
                xl = o.get_layerOutput(l,xl);
            end
            yhat = xl;
            err.MSE = mse(yhat',y_test);
            err.RMSE = sqrt(err.MSE);
            STD = std(y_test);
            err.NDEI = err.RMSE / STD;

        end

        %% MAIN LOOP
        function [o,yhat] = main(o,Xtrain,Ytrain,epoch)
                yhat = inf(size(Ytrain));
                for k = 1:o.K
                    % k
                    xk = Xtrain(k,:)';
                    [yk, o] = o.forward(xk,k,epoch);
                    yhat(k,:) = yk;
                    o = o.backward(yk, Ytrain(k,:)');
                end
                % MSE = mse(Ytrain, yhat);
        end
    end

    methods (Access=private)
        

        %% FORWARD
        function [yk, o] = forward(o, xl, k, epoch)
            for l = 1:o.L
                Ml = o.M(l);
                Nl = o.N(l);

                if epoch == 1 && k == 1 % Stage(0): Initialization
                    o.gMu(1:Ml,l) = xl;
                    o.gX(l) = o.squEucNorm(xl);
                    o = o.ini(xl,l); % update o.N(l)
                else  % Stage(1)
                    o.gX(l) = o.gX(l) + (o.squEucNorm(xl) - o.gX(l))/k;
                    o.gMu(1:Ml,l) = o.gMu(1:Ml,l) + (xl - o.gMu(1:Ml,l))/k;
                    Dl = o.get_density(xl,1:Nl,l); % less distance has more density
                    o.lambda(Nl,l) = Dl(Nl) / sum(Dl);
                    [max_dens,n_star] = max(Dl);
                    if max_dens < o.delta
                        o = o.ini(xl,l);   % update o.N(l)
                    else
                        o = o.updateRule(xl,n_star,l);
                    end
                end
                o.xb(2:Ml+1,l) = xl;
                xll = o.get_layerOutput(l,xl);
                if sum(isnan(xll))
                    error('h')
                end
                xl=xll;
            end
            yk = xl;    % last output of layers
        end

        %% INITIALIZATION
        function o = ini(o,xl,l)
            o.N(l) = o.N(l) + 1;        % add new rule/cluster
            Nl = o.N(l);     % new rule of layer(l)
            Ml = o.M(l);    % number of layer input
            Wl = o.W(l);    % number of layer output

            o.P(1:Ml,Nl,l) = xl;
            o.A(1:Wl,1:Ml+1,Nl,l) = randi([0 1],Wl,Ml+1) / (Ml+1);
            o.Cc(1:Ml,Nl,l) = xl;
            o.CX(Nl,l) = o.squEucNorm(xl);
            o.CS(Nl,l) = 1;
            o.lambda(Nl,l) = 1;
        end

        %% DETERMIN LAYER OUTPUT
        function yl = get_layerOutput(o,l,xl)
            % xl : (#feature,#instance)
            xbar = [ones(1,size(xl,2)); xl];
            
            % y_l = sum(lam_nl * y_nl)
            % y_nl = AF(A_nl * xbar_l)
            lam_ynl = inf(o.W(l), size(xl,2), o.N(l));
            for n = 1:o.N(l)
                lam_ynl(:,:,n) = o.lambda(n,l) * o.AF(o.A(1:o.W(l),1:o.M(l)+1,n,l) * xbar);
            end
            yl = sum(lam_ynl,3);
        end

        %% UPDATE RULE FUNCTION
        function o = updateRule(o,xl,n_star,l)
            Ccn_star = o.Cc(1:o.M(l), n_star, l);
            CXn_star = o.CX(n_star, l);

            o.CS(n_star, l) = o.CS(n_star, l) + 1;
            o.Cc(1:o.M(l),n_star,l) = Ccn_star + (xl - Ccn_star) / o.CS(n_star, l);
            o.CX(n_star, l) = CXn_star + (o.squEucNorm(xl) - CXn_star) / o.CS(n_star, l);
        end

        %% BACKPROPAGATION
        function o = backward(o, yk, rk)
            DeDy = yk - rk;  % (wl,1)
            % ek = DeDy' * DeDy / 2;

            % INITIALIZE SOME REPEATED VARIABLES
            DlamDx = inf(max(o.M),o.K,o.L);  %(max_input, max_rule, max_layer)
            AF_Axb = inf(max(o.W),o.K,o.L);   %(max_output, max_rule, max_layer)
            AFp = inf(max(o.W),o.K,o.L);   %(max_output, max_rule, max_layer)
            for l = 1:o.L
                m = 1:o.M(l);
                mxb = 1:o.M(l)+1;
                mx = 2:o.M(l)+1;
                w = 1:o.W(l);
                n = 1:o.N(l);

                % 1.    DlamDx_nl =  lam_nl * ( 2(p_nl - x_l)/taw_nl^2 - sum_i:Nl(lam_il * 2(p_il - x_l)/taw_il^2) )
                taw_nl = o.get_taw(n,l);
                DlamDx(m,n,l) = o.lambda(n,l)' .* (...
                    (2*(o.P(m,n,l) - o.xb(mx,l)) ./ taw_nl'.^2) - ...
                    sum(o.lambda(n,l)' .* (2*(o.P(m,n,l) - o.xb(mx,l)) ./ taw_nl'.^2), 2));

                % 2.    Axb_nl = A_nl * xb_l
                Al = o.A(w,mxb,n,l);
                xbl = o.xb(mxb,l);
                for nn = n
                    AF_Axb(w,nn,l) = Al(w,mxb,nn) * xbl;
                end
                AF_Axb(w,n,l) = o.AF(AF_Axb(w,n,l));
                clear Al xbl

                % 3.    AFprim_nl = AF(Axb_nl) .* (1 - AF(Axb_nl))
                AFp(w,n,l) = AF_Axb(w,n,l) .* (1 - AF_Axb(w,n,l));
            end

            % d (BACKWARD)
            o.d(1:o.W(o.L),o.L) = DeDy; % d of last layer
            for l = o.L-1 : -1 : 1
                m1 = 1:o.M(l+1);
                mx1 = 2:o.M(l+1)+1;
                w = 1:o.W(l);
                w1 = 1:o.W(l+1);

                for n = 1:o.N(l+1)
                    o.d(w,l) = o.d(w,l) + ...
                    DlamDx(m1,n,l+1) * AF_Axb(w1,n,l+1)' * o.d(w1,l+1) + ...
                    o.lambda(n,l+1) * o.A(w1,mx1,n,l+1)' * ( o.d(w1,l+1) .* AFp(w1,n,l+1) );
                end
            end

            for l = 1:o.L
                w = 1:o.W(l);
                
                n = 1:o.N(l);
                mxb = 1:o.M(l)+1;
                dedA = pagemtimes(reshape(o.lambda(n,l)' .* (o.d(w,l) .* AFp(w,n,l)),[],1,o.N(l)), o.xb(mxb,l)' );
                o.A(w,mxb,n,l) = o.A(w,mxb,n,l) - o.alpha * dedA;
            end
        end

        %% LOCAL DENSITY FUNCTION
        function D = get_density(o,xl,n,l)
            % xl : (#featre,#instance)
            D = exp(- o.squEucNorm(xl - o.P(1:o.M(l),n,l)) ./ o.get_taw(n,l).^2);
        end

        %% ACTIVATION FUNCTION
        function out = AF(o,in)
            % switch o.ActivationFunction
            %     case "sigmoid"
                    out = 1 ./ (1 + exp(-in));
            %     case "relu"
            %         in(in<0) = 0;
            %         out = in;
            % end  
        end

        %% TAW FUNCTION
        function taw = get_taw(o,n,l)
            % if ~isreal((o.gX(l) - o.squEucNorm(o.gMu(1:o.M(l),l)) + o.CX(n,l) - o.squEucNorm(o.Cc(1:o.M(l),n,l)))/2)
            %     0;
            % end
            % taw = sqrt(( (o.gX(l) - o.squEucNorm(o.gMu(1:o.M(l),l))) + (o.CX(n,l) - o.squEucNorm(o.Cc(1:o.M(l),n,l))) )/2);
            % if sum(~isreal(taw))
            %     warning("complex taw")
            % end            
            taw = sqrt(( abs(o.gX(l) - o.squEucNorm(o.gMu(1:o.M(l),l))) + abs(o.CX(n,l) - o.squEucNorm(o.Cc(1:o.M(l),n,l))) )/2);
            taw(taw == 0) = eps;
        end

        %% SQUARED EUCLIDEAN NORM
        function out = squEucNorm(~,x)
            sq = sqrt(pagemtimes(pagetranspose(x),x));
            out = zeros(size(x,2),size(x,3));
            for i = 1:size(x,3)
                out(:,i) = diag(sq(:,:,i)).^2;
            end
        end

    end

end
