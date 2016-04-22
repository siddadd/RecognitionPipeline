% Y = labels
% X = data
function [C,G] = nonlinear_parametric_search(Y,X)

bestcv = 0;
for log2g = 1:3    
    for log2c = -1:3    
        cmd = ['-q -s 0 -t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(Y, X, cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g; 
        end
        fprintf('%g %g %g (best c=%g, best g =%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);  
    end
end

C = bestc;
G = bestg; 