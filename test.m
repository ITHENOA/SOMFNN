clear
clc
% load mgdata.dat
% time = mgdata(:,1);
% x = mgdata(:, 2);
% for t = 118:1117 
%     Data(t-117,:) = [x(t-18) x(t-12) x(t-6) x(t) x(t+6)]; 
% end
% Xtr = Data(1:800, 1:end-1);
% Ytr = Data(1:800, end);
% Xte = Data(801:end, 1:end-1);
% Yte = Data(801:end, end);
% clear Data x t mgdata time

load mackey_0.1.mat
for k = 201:3200
    Xtr(k-200,:) = [X(k-18) X(k-12) X(k-6)];
    Ytr(k-200,:) = X(k);
end
for k = 5001:5500
    Xte(k-5000,:) = [X(k-18) X(k-12) X(k-6)];
    Yte(k-5000,:) = X(k);
end
clear X k

tic
% net = MSOFNN(X_train,Y_train,3,[3 3 1]).train
toc
% data
tic
% net2=MSOFNNplus(X_train,Y_train,3,"batchSize",1,"MaxEpoch",100)
toc
% idx=randperm(398);
% Xtr = data(idx(1:196),1:6);
% Xte = data(idx(197:end),1:6);
% Ytr = data(idx(1:196),7);
% Yte = data(idx(197:end),7);
tic
% for i = 1:6
net3=MSOFNNplus(Xtr,Ytr,3,...
    "batchSize",32,...
    "MaxEpoch",100,...
    "DensityThreshold",exp(-3),...
    "verbose",1,...
    "LearningRate",1)

[~,err] = net3.test(Xte,Yte)
% err(i) = net3.MSE_report.Mean;
% end
% mean(err)
toc