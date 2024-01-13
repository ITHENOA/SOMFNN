classdef Similarity
    methods (Static)

        % Euclidian distance
        function S = euc(A,B)
            S = norm(A-B);
        end

        % Cosine
        function S = cos(A,B)
            S = dot(A,B) / norm(A) / norm(B);
        end

        % dot Product
        function S = dotProd(A,B)
            S = dot(A,B);
        end

        % doi.org/10.1007/s41066-023-00366-1
        function S = kumar(A,B)
            n = size(A,2);
            S = sum(1 ./ (1 + abs(A-B))) / n;
        end
    end
end