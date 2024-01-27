classdef myclass
    properties
        arg1
        arg2
        optimizer
    end
    
    methods
        % function obj = myclass(arg1, arg2, varargin)
        %     p = inputParser;
        %     addRequired(p, 'arg1', @isnumeric);
        %     addRequired(p, 'arg2', @isnumeric);
        %     addParameter(p, 'optimizer', 'sgdm', @(x) ismember(x, {'sgdm', 'adam', 'rmsprop'}));
        % 
        %     parse(p, arg1, arg2, varargin{:});
        % 
        %     obj.arg1 = p.Results.arg1;
        %     obj.arg2 = p.Results.arg2;
        %     obj.optimizer = p.Results.optimizer;
        %     obj = myclass(5, 55, 'optimizer', '
        % end
        function obj = myclass(varargin)
            % Support name-value pair arguments when constructing object
            setProperties(obj,nargin,varargin{:})
        end

    end
end
