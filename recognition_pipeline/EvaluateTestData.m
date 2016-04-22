function ResultData = EvaluateTestData(FeatureData, ClassifierModel, params, kfoldIter)

ResultData = cell(length(params.featuretype),1);

for feat = 1:length(params.featuretype)
    
    featureName = char(params.featuretype{feat});
    
    if exist([params.resultpath featureName, '/kfold', int2str(kfoldIter) '/' params.classifiertype 'result', params.resultfiletype],'file')
        if params.verbose
            fprintf([params.classifiertype ' model for ' featureName ' Results Exists\n']);
        end
        load([params.resultpath featureName,'/kfold', int2str(kfoldIter), '/', params.classifiertype 'result', params.resultfiletype], 'resultData');
        ResultData{feat} = resultData;   
        fprintf('Accuracy is %f\n', ResultData{feat}.accuracy(1));
    else
        if params.verbose
            fprintf(['Evaluate ' featureName ' BoW using ' params.classifiertype '\n']);
        end
        
        if ~exist([params.resultpath featureName,'/kfold', int2str(kfoldIter)], 'dir')
            mkdir([params.resultpath featureName,'/kfold', int2str(kfoldIter)]);
        end
        
        X = []; Y = [];
        
        % Get Test Features
        for i = 1:numel(FeatureData{feat})

            dirs = FeatureData{feat}(i);
            className = {dirs([dirs.isdir]).name};
            className = setdiff(className, {'.','..'});
            if isempty(className)
                continue
            end
            
            files = dir([params.testfeaturepath, featureName, '/kfold', int2str(kfoldIter),'/',char(className),'/*', params.featurefiletype]);
            
            for j = 1 : numel(files)
                
                % Load features
                load([params.testfeaturepath, featureName, '/kfold', int2str(kfoldIter),'/',char(className),'/',files(j).name], 'fvector');
                x = fvector; 
                xlabel = i;
                X = [X x(:)];
                Y = [Y xlabel];
            end
        end
              
        X = X';
        Y = Y';
        % Save as mat as well as in libsvm format to train independently if
        % needed
        save( [params.resultpath featureName,'/kfold', int2str(kfoldIter), '/test_data' params.resultfiletype], 'X', 'Y', '-v7.3');        
        libsvmwrite([params.resultpath featureName,'/kfold', int2str(kfoldIter), '/', 'test_data.txt'], Y, sparse(double(X)));
        
        if strcmp(params.classifiertype, 'LinearSVM')        
            [test_y, test_x] = libsvmread([params.resultpath featureName,'/kfold', int2str(kfoldIter), '/', 'test_data.txt']);
            [predLabels, accuracy, decisionValues] = svmpredict(test_y, test_x, ClassifierModel{feat});
        elseif strcmp('NonlinearSVM', params.classifiertype)
            [test_y, test_x] = libsvmread([params.resultpath featureName,'/kfold', int2str(kfoldIter), '/', 'test_data.txt']);
            [predLabels, accuracy, decisionValues] = svmpredict(test_y, test_x, ClassifierModel{feat});
        end               
            
        resultData.predLabels = predLabels;
        resultData.gtLabels = test_y;
        resultData.accuracy = accuracy; 
        resultData.decisionValues = decisionValues;
        
        save( [params.resultpath featureName,'/kfold', int2str(kfoldIter), '/', params.classifiertype 'result', params.resultfiletype] , 'resultData', '-v7.3');
        ResultData{feat} = resultData;
    end
    
end
