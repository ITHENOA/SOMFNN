function opt = msofnnOption(n_Layer,opt)
arguments
    n_Layer (1,1) {mustBeInteger,mustBePositive}
    opt.n_hiddenNodes (1,:) {mustBeInteger,mustBePositive,mustBeVector} = "Auto"
    opt.LearningRate (1,1) {mustBeInteger,mustBePositive} = 0.001
    opt.MaxEpoch (1,1) {mustBeInteger,mustBePositive} = 500
    opt.DensityThreshold (1,1) {mustBeInteger,mustBePositive} = exp(-3)
    opt.verbose (1,1) {logical} = 0
    opt.Plot (1,1) {logical} = 0
    opt.ActivationFunction {mustBeTextScalar} = "Sigmoid"
    opt.BatchNormType {mustBeTextScalar} = "none"
    opt.SolverName {mustBeTextScalar} = "SGD"
    opt.adampar_epsilon = 1e-8
    opt.adampar_beta1 = 0.9
    opt.adampar_beta2 = 0.999
    opt.adampar_m0 = 0
    opt.adampar_v0 = 0
end
opt.ActivationFunction = validatestring(opt.ActivationFunction,{"Sigmoid","ReLU","LeakyRelu","Tanh","ELU"}); % linear
opt.BatchNormType = validatestring(opt.BatchNormType,{"none","zscore"});
opt.SolverName = validatestring(opt.SolverName,{"SGD","Adam","MiniBatchGD"});
end