% calculateSum.m

function [result] = calculateSum(varargin)
    % calculateSum - Custom function to calculate the sum of input values.
    %
    % Function Signature 1:
    %   result = calculateSum(value1, value2, ...)
    %
    % Function Signature 2:
    %   [result] = calculateSum(value1, value2, ..., param)
    %
    % Function Signature 3:
    %   [result] = calculateSum(value1, value2, ..., 'param1', value1, 'param2', value2, ...)

    result = sum([varargin{:}]);
end
