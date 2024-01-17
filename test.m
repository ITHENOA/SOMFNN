clear
clc
load mgdata.dat
time = mgdata(:,1);
x = mgdata(:, 2);
for t = 118:1117 
    Data(t-117,:) = [x(t-18) x(t-12) x(t-6) x(t) x(t+6)]; 
end
X_train = Data(1:800, 1:end-1);
Y_train = Data(1:800, end);
X_test = Data(801:end, 1:end-1);
Y_test = Data(801:end, end);
clear Data x t mgdata time

net = MSOFNN(X_train,Y_train,5,[6 8 6 4])