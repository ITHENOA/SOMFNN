classdef HelpFunc
    methods (Static)
        function fis = fixRange(fis,data)
            fis.Inputs(1).Range = [min(data(:,1)) max(data(:,1))];
            fis.Inputs(2).Range = [min(data(:,2)) max(data(:,2))];
            fis.Outputs.Range = [0 6];
        end
        function [x1,x2,y] = getpars(fis)
            [x1.par1, x1.par2, x1.par3, x1.par4, x1.par5] = fis.Input(1).MembershipFunctions.Parameters;
            [x2.par1, x2.par2, x2.par3, x2.par4, x2.par5] = fis.Input(2).MembershipFunctions.Parameters;
            [y.par1, y.par2, y.par3, y.par4, y.par5] = fis.Outputs.MembershipFunctions.Parameters;
        end
    end
end