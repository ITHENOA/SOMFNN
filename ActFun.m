classdef ActFun
    methods (Static)
        function [y,yp] = Linear(x,a,c)
            % Pros
            % – Provides a range of activations, not binary.
            % – Can connect multiple neurons and make decisions based on max activation.
            % – Constant gradient for stable descent.
            % – Changes are constant for error correction.
            % Cons
            % – Limited modeling capacity due to linearity.
            arguments
                x
                a = 1
                c = 0
            end
            y = a * x + c;
            yp = a*ones(size(x));
        end

        function [y,yp] = ReLU(x)
            % Pros
            % – Solves the vanishing gradient problem.
            % – Computationally efficient.
            % Cons
            % – Can lead to “Dead Neurons” due to fragile gradients. Should be used only in hidden layers.
            y = max(x,0);
            yp = x;
            yp(yp >= 0) = 1;
            yp(yp < 0) = 0;
        end

        function [y,yp] = LeakyRelu(x,alpha)
            % Pros
            % – Mitigates the “dying ReLU” problem with a small negative slope.
            % Cons
            % – Lacks complexity for some classification tasks.
            arguments
                x
                alpha = 0.01;
            end
            y = max(alpha * x, x);
            yp = x;
            yp(yp >= 0) = 1;
            yp(yp < 0) = alpha;
        end

        function [y,yp] = Sigmoid(x)
            % Pros
            % – Nonlinear, allowing complex combinations.
            % – Produces analog activation, not binary.
            % Cons
            % – Suffers from the “vanishing gradients” problem, making it slow to learn.
            y = 1 ./ (1 + exp(-x));
            yp = y .* (1 - y);
        end

        function [y,yp] = Tanh(x)
            % Pros
            % – Stronger gradient compared to sigmoid.
            % – Addresses the vanishing gradient issue to some extent.
            % Cons
            % – Still prone to vanishing gradient problems.
            % y = 2 ./ (1 - exp(-2 * x)) - 1;
            y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
            yp = 1 - y.^2;
        end

        function [y,yp] = ELU(x,alpha)
            % Pros
            % – Can produce negative outputs for x>0.
            % Cons
            % – May lead to unbounded activations, resulting in a wide output range.
            arguments
                x
                alpha = 0.01;
            end
            y = max(alpha * (exp(x) - 1), x);
            yp = x;
            yp(yp >= 0) = 1;
            yp(yp < 0) = y + alpha;
        end

        function [y,yp] = Softmax(x)
            e = exp(x);
            s = sum(e); 
            y = e ./ s;
            yp = 1 ./ (s-e);
        end
    end
end