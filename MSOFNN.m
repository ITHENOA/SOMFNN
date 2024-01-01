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
        xbar    % [1;xkl]
        lambda  % Firing strength of "xl" to "Rnl"
        D       % Local density of "xkl" at each cluster
        taw     % Width of the exponential kernel associated with "Rnl"
    end
    methods
        function o = MSOFNN(X_train,y_train,L,delta,w)
            arguments
                X_train %(#instance, #dim/feature)
                y_train %(#instance, #output) : if (#output>1) then we have MIMO system
                L (1,1) {mustBePositive} = 2 % number of layer
                delta (1,1) {mustBePositive} = exp(-2) % thereshold
                w (1,:) {mustBePositive} = ones(1,L-1)*3*size(y_train,2) % (1,L-1) or (1,1) : number of output of each layer exept last one
            end

            % construct W and M
            if numel(w) == L - 1
                o.W = [w,size(y_train,2)]; %(1,l)
            elseif numel(w) == L
                o.W = w;
            else
                error("incorrect number of w's elements")
            end
            o.M = [size(X_train,2), o.W(1:end-1)]; %(1,l)

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

            o.K = size(X_train, 1);     % Number of instances
            max_input = max(o.W);       % Maximum input of layers
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
            o.xbar = inf(max_input+1, max_layer);                   % xbar(Ml+1,1)
            o.lambda = inf(max_rule, max_layer);                    % lambda(n,l)              
            o.D = inf(max_rule, max_layer);                         % D(n,l)
            o.taw = inf(max_rule, max_layer);                       % taw(n,l)

            % Main Loop
            o.xbar(1,:) = ones(1,max_layer);
            for k = 1:o.K
                xk = X_train(k,:)';
                yk = o.main(xk,k);
                o.A = o.backProp(yk,y_train(k,:)');
            end

        end

        %% Main Loop
        function yk = main(o,xl,k)
            for l = 1:L
                Ml = o.M(l);
                Wl = o.W(l);
                Nl = o.N(l);

                if k == 1 % Stage(0): Initialization
                    o.gMu(1:Ml,l) = xl;
                    o.gX(l) = o.squEucNorm(xl);
                    o = o.ini(xl,l);
                    Nl = o.N(l);
                else  % Stage(1)
                    o.gX(l) = o.gX(l) + (o.squEucNorm(xl) - o.gX(l))/k;
                    o.gMu(1:Ml,l) = o.gMu(1:Ml,l) + (xl - o.gMu(1:Ml,l))/k;
                    % test
                    % d1 = o.densFunc(o.P(1:o.M(l),Nl,l)+0.5,l,1:Nl)
                    % d2 = o.densFunc(o.P(1:o.M(l),Nl,l)+1,l,1:Nl)
                    Dl = o.get_density(xl,l,1:Nl); % less distance has more density
                    o.lambda(Nl,l) = Dl(Nl) / sum(Dl);
                    [max_dens,n_star] = max(Dl);
                    if max_dens < 1
                        o = o.ini(xl,l);
                        Nl = o.N(l);
                    else
                        o = o.updateRule(xl,n_star,l);
                    end
                end
                o.xbar(2:Ml+1,l) = xl;
                xl = o.get_layerOutput(l, o.lambda(1:Nl,l), o.A(1:Wl,1:Ml+1,1:Nl,l), o.xbar(1:Ml+1,l));
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

        %% Determin Layer Output
        function yl = get_layerOutput(o,l,lambda,A,xbar)
            ynl = inf(o.W(l),o.N(l));

            for Nl = 1:o.N(l)
                ynl(:,Nl) = lambda(Nl) * o.AF(A(:,:,Nl) * xbar);
            end
            yl = sum(ynl,2);
        end

        %% update rule function
        function o = updateRule(o,xl,n_star,l)
            Ccn_star = o.Cc(1:o.M(l), n_star, l);
            CXn_star = o.CX(n_star, l);

            o.CS(n_star, l) = o.CS(n_star, l) + 1;
            o.Cc(1:o.M(l),n_star,l) = Ccn_star + (xl - Ccn_star) / o.CS(n_star, l);
            o.CX(n_star, l) = CXn_star + (o.squEucNorm(xl) - CXn_star) / o.CS(n_star, l);
        end
        
        %% BackPropagation
        function A = backProp(o,yk,rk)
            dedy = yk - rk;  % (wl,1)
            ek = dedy' * dedy / 2;

            % A = o.A - alpha * dedA;
        end

        %% local density function
        function D = get_density(o,xl,l,n)
            D = exp(- o.squEucNorm(xl - o.P(1:o.M(l),n,l)) ./ o.get_taw(l,n));
        end

        %% Activation Function
        function out = AF(~,in)
            out = 1 ./ (1 + exp(-in));
        end

        %% taw function
        function taw = get_taw(o,l,n)
            taw = sqrt((o.gX(l) - o.squEucNorm(o.gMu(1:o.M(l),l)) + o.CX(n,l) - o.squEucNorm(o.Cc(1:o.M(l),n,l)))/2);
        end

        %% Squared Euclidean Norm
        function out = squEucNorm(~,x)
            out = inf(size(x,2),1);

            parfor i = 1:size(x,2)
                out(i) = sqrt(x(:,i)' * x(:,i));
            end
        end

    end

end
