function fis1_tuned=tunefis_k(fis1,input_tr,output_tr)

[in,out,~] = getTunableSettings(fis1);
opt = tunefisOptions("Method","ga");
opt.MethodOptions.MaxGenerations = 20;
% opt.KFoldValue = 3;
% opt.ValidationWindowSize = 5;
% opt.ValidationTolerance = 0.05;
fis1_tuned = tunefis(fis1,[in;out],input_tr,output_tr,opt);
end