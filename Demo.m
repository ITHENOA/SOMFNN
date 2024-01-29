% clear all
% clc
% close all
N=10;
label_est = []
for ii=1:1:N
    % load(['data_' num2str(ii) '.mat'])
    %% Vectorizing labels of training data
    % X=full(ind2vec(LTra1')');
    % X(X==0)=-1;
    %% Training
    Input.x=Xtr;Input.y=Ytr;
    tic
    [Output]=SAFLS(Input,'L');
    texe(ii)=toc;
    %% Testing
    Input.x=Xte;Input.Syst=Output.Syst;
    [Output]=SAFLS(Input,'T');
    mse(Output.Ye,Yte)
    % label_est(ii)=Output.Ye;
    % [~,label_est]=max(label_est,[],2);
    % Acc(ii)=sum(sum(confusionmat(Yte,label_est).*(eye(length(unique(Yte))))))/length(Yte);
end
% [mean(Acc),std(Acc),max(Acc)]
% [mean(texe),std(texe),max(texe)]
mse(label_est,Yte)