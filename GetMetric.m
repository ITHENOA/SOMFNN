function metrics = GetMetric(type,yh,y,get_all)
type = validatestring(type,["classification","regression"]);
switch type
    case "classification"
        % Mean Squared Error (MSE)
        metrics.MSE = mse(yh,y);
        if strcmpi(get_all,"all")
            % Root Mean Squared Error (RMSE)
            metrics.RMSE = sqrt(metrics.MSE);
            % Mean Absolute Error (MAE)
            metrics.mae = mae(y,yh);
            % R-squared (R²)
            metrics.R2 = 1 - (sum((y - yh).^2) / sum((y - mean(yh)).^2));
            % Mean Absolute Percentage Error (MAPE)
            metrics.MAPE = mean(abs((y - yh) ./ y) * 100);
            % Explained Variance Score
            metrics.explained_variance = 1 - (var(y - yh) / var(y));
            % Explained Variance Score
            metrics.explainedVariance = mean(1 - (sum((y - yh).^2) ./ sum((y - mean(yh)).^2)));
            % Mean Squared Logarithmic Error (MSLE)
            metrics.msle = mean( mean((log1p(y) - log1p(yh)).^2));
            % Mean Bias Deviation (MBD)
            metrics.mbd = mean( mean(y - yh));
            % Normalized Root Mean Squared Error (NRMSE)
            range = max(y) - min(y);
            metrics.nrmse = mean( metrics.rmse ./ range);
            % Mean Percentage Error (MPE)
            metrics.mpe =  mean(mean((y - yh) ./ y) * 100);
            % Mean Absolute Percentage Error (MAPE)
            metrics.mape =  mean(mean(abs((y - yh) ./ y)) * 100);
        end

    case "regression"
        cm = confusionmat(y, yh);
        nClass = size(cm,1);
        [TP,FP,FN,TN] = deal(zeros(1,nClass));
        for i = 1:nClass
            TP(i) = cm(i,i);
            FP(i) = sum(cm(:, i), 1) - TP(i);
            FN(i) = sum(cm(i, :), 2) - TP(i);
            TN(i) = sum(cm(:)) - TP(i) - FP(i) - FN(i);
        end
        % Accuracy; [0,1]
        metrics.ACC = sum(TP) / sum(cm,"all");
        if strcmpi(get_all,"all")
            % Precision; [0,1]
            metrics.PREC = sum(TP) / (sum(TP) + sum(FP)); % macro = mean(TP ./ (TP + FP));
            % Recall; [0,1]
            metrics.RECALL = sum(TP) / (sum(TP) + sum(FN)); % macro = mean(TP ./ (TP + FN));
            % F1 Score; [0,1]
            metrics.F1SCORE = 2 * (metrics.PREC .* metrics.RECALL) / (metrics.PREC + metrics.RECALL);
            % Cohen's Kappa
            metrics.KAPPA = mean(2*(TN.*TP - FN.*FP)./(TP+FP).*(TN+FP)+(TP+FN).*(TN+FN));
            % Matthews Correlation Coefficient (MCC);  [−1,1]
            metrics.MCC = mean((TN.*TP - FN.*FP)./sqrt((TP+FP).*(TP+FN).*(TN+FP).*(TN+FN)));
        end
    otherwise
        error("Invalid 'ProblemType'.")
end
end