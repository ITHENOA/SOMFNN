classdef Eval
    properties
        y
        yhat
        n
    end
    methods
        function o = Eval(y,yhat)
            o.y = y;
            o.yhat = yhat;
            o.n = size(y,1);
        end
        function MSE = MSE(o)
            MSE = sum((o.y - o.yhat).^2)/o.n;
        end
    end
end
