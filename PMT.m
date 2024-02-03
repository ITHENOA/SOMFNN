function out = PMT(A,B,C)
if nargin == 2
    out = pagemtimes(A,B);
elseif nargin == 3
    out = pagemtimes(pagemtimes(A,B),C);
end