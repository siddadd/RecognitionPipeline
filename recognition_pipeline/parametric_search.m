function C = linear_parametric_search(Y,X)

bestcv = 0;
for log2c = -1:3
    log2g = 'null';
    cmd = ['-q -s 0 -t 0 -v 5 -c ', num2str(2^log2c), ' -g ', 'null'];
    cv = svmtrain(Y, X, cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c;
    end
    fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);  
end

C = bestc;