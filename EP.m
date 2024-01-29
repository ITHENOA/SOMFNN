function error_fis1_tuned=EP(fis1_tuned,input_ts,output_ts,sg)
figure
gensurf(fis1_tuned)

output_fis1_tuned = evalfis(fis1_tuned,input_ts);
error_fis1_tuned=mean(abs(output_fis1_tuned-output_ts));

figure
subplot(2,1,1)
plot(1:length(input_ts),output_fis1_tuned,"LineWidth",2)
grid on
hold on
plot(1:length(input_ts),output_ts,"LineWidth",2)
xlabel('Input Index')
ylabel('Output')
title('Approximation vs Reference')
legend('Approximation','Reference')

subplot(2,1,2)
plot(1:length(input_ts),abs(output_fis1_tuned-output_ts),"LineWidth",2)
grid on
xlabel('Input Index')
ylabel('Error')
title('Error Calculation')


figure
plotfis(fis1_tuned)
figure
plotmf(fis1_tuned,'input',1)
figure
plotmf(fis1_tuned,'input',2)
if sg==0
    figure
    plotmf(fis1_tuned,'output',1)
end
end