function [dataTr, dataTe, dataTe2] = datasetBenchmark(datasetName,datasetDirectory,kfold)
if nargin<3, kfold=0; end
% datasetName:
%   regression = [sp500, mglass, ampg]
%   classification = [pen, gesphase, liver, kmg]
% datasetDirectory: folder of data

addpath(datasetDirectory) % "./dataset"
switch lower(datasetName)
    case "pen"
        % Pen (10 class)
        data = readmatrix("pen_class10.xlsx");
        X = data(:,1:end-1);
        Y = data(:,end);
        idx = randperm(numel(Y));
        dataTr.x = X(idx(1:7494),:);
        dataTr.y = Y(idx(1:7494),1);
        dataTe.x = X(idx(7495:end),:);
        dataTe.y = Y(idx(7495:end),1);

    case "gesphase"
        % GesPhase (5 class)
        load GesturePhase_class5.mat GestrurePhase
        GestrurePhase(logical(sum(isnan(GestrurePhase),2)),:) = [];
        X = GestrurePhase(:,1:end-1);
        Y = GestrurePhase(:,end);
        if kfold
            [dataTr,dataTe] = kFold(X,Y,kfold);
        else
            idx = randperm(numel(Y));
            dataTr.x = X(idx(1:6500),:);
            dataTr.y = Y(idx(1:6500),1);
            dataTe.x = X(idx(6501:end),:);
            dataTe.y = Y(idx(6501:end),1);
        end

    case "sp500"
        % SP500 (1 reg)
        data = readmatrix("SP500_reg.xlsx");
        data = data(:,5);
        data = [data ; flipud(data)];
        data = normalize(data,1,"range");
        for t = 5:numel(data)-1
            X(t,:) = [data(t-4) data(t-3) data(t-2) data(t-1) data(t)];
            Y(t,1) = data(t+1);
        end
        dataTr.x = X(1:15038,:);
        dataTr.y = Y(1:15038,1);
        dataTe.x = X(15039:end,:);
        dataTe.y = Y(15039:end,1);

    case "liver"
        % liver (Binery classification)
        data = readmatrix("liver_glass2.xlsx");
        data = shuffle(data);
        Y = data(:,end);
        X = data(:,1:end-1);
        if kfold
            [dataTr,dataTe] = kFold(X,Y,kfold);
            % idx = randperm(size(data,1));
            % dataTe2.x = X(idx(201:end),:);
            % dataTe2.y = Y(idx(201:end));
            % [dataTr,dataTe] = kfold(X(idx(1:200),:),Y(idx(1:200),:),kfold);
        else
            idx = randperm(size(data,1));
            idxTr = idx(1:200);
            idxTe = idx(201:end);
            dataTr.x = X(idxTr,:);
            dataTr.y = Y(idxTr);
            dataTe.x = X(idxTe,:);
            dataTe.y = Y(idxTe);
        end

    case "kmg"
        % KMG (3 class)
        addpath(datasetDirectory + "\KMG")
        load fieldAmp.mat fieldAmp
        data = fieldAmp;
        idx = randperm(size(data,1));
        dataTr.x = data(idx(1:10000),1:end-1);
        dataTr.y = data(idx(1:10000),end);
        dataTe.x = data(idx(10001:end),1:end-1);
        dataTe.y = data(idx(10001:end),end);

    case "mglass"
        % Mackey Glass (1 reg)
        load MackeyGlassNew.mat data
        data(:,1) = [];
        data = normalize(data,1,"range");
        for t = 201:3200
            dataTr.x(t-200,:) = [data(t-18) data(t-12) data(t-6) data(t)];
            dataTr.y(t-200,:) = data(t + 85);
        end
        for t = 5001:5500
            dataTe.x(t-5000,:) = [data(t-18) data(t-12) data(t-6) data(t)];
            dataTe.y(t-5000,:) = data(t + 85);
        end

    case "ampg"
        % Auto MPG (1 reg)
        data = readmatrix("autoMPG_reg.xlsx");
        data(logical(sum(isnan(data),2)),:) = [];
        idx=randperm(size(data,1));
        dataTr.x = data(idx(1:196),1:7);
        dataTr.y = data(idx(197:end),1:7);
        dataTe.x = data(idx(1:196),8);
        dataTe.y = data(idx(197:end),8);

end

