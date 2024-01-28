function testFunc(varargin)
[varargin{:}] = convertStringsToChars(varargin{:});
pnames = {   'distance'  'start' 'replicates' 'emptyaction' 'onlinephase' 'options' 'maxiter' 'display'};
dflts =  {'sqeuclidean' 'plus'          []  'singleton'         'off'        []        []        []};
[distance,start,reps,emptyact,online,options,maxit,display] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});
distNames = {'sqeuclidean','cityblock','cosine','correlation','hamming'};
distance = internal.stats.getParamVal(distance,distNames,'''Distance''');
switch distance
    case 'cosine'
        Xnorm = sqrt(sum(X.^2, 2));
        if any(min(Xnorm) <= eps(max(Xnorm)))
            error(message('stats:kmeans:ZeroDataForCos'));
        end
        X =  X./Xnorm;
    case 'correlation'
        X = X - mean(X,2);
        Xnorm = sqrt(sum(X.^2, 2));
        if any(min(Xnorm) <= eps(max(Xnorm)))
            error(message('stats:kmeans:ConstantDataForCorr'));
        end
        X =  X./Xnorm;
     case 'hamming'
       if  ~all( X(:) ==0 | X(:)==1)
            error(message('stats:kmeans:NonbinaryDataForHamm'));
      end
end