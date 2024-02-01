% Initialization
tau = 17;            % Time delay
x0 = 1.2;            % Initial condition
N = 5500;            % Total number of steps to simulate
x = zeros(N, 1);     % Preallocate the array for speed
x(1:tau+1) = x0;     % Initialize the first tau+1 values to the initial condition

% Parameters for the Mackey-Glass equation
a = 0.2;
b = 0.1;
n = 10;              % Power of x in the equation

% Simulation using Euler's method
for k = tau+1:N-1
    x(k+1) = x(k) + (a  x(k-tau) / (1 + x(k-tau)^n) - b  x(k));
end

% Extract training samples from the time period between k = 201 and k = 3200
training_indices = 201:3200;
training_samples = x(training_indices);

% Extract testing samples from the time period between k = 5001 and k = 5500
testing_indices = 5001:5500;
testing_samples = x(testing_indices);

% Display the results
plot(x);
title('Mackey-Glass Time Series');
xlabel('Time steps');
ylabel('x_k');
