function ModelData = TrainClassifier(FeatureData, params, kfoldIter)

ModelData = cell(length(params.featuretype),1);

for feat = 1:length(params.featuretype)
    
    featureName = char(params.featuretype{feat});
    
    if exist([params.modelpath featureName, '/kfold', int2str(kfoldIter), '/', params.classifiertype 'model', params.modelfiletype], 'file')
        if params.verbose
            fprintf([featureName ' ' params.classifiertype ' Classifier Exists\n']);
        end
        load([params.modelpath featureName,'/kfold', int2str(kfoldIter), '/', params.classifiertype 'model', params.modelfiletype], 'model');
        ModelData{feat} = model;
    else
        if params.verbose
            fprintf(['Build ' params.classifiertype ' Classifier for ' featureName '\n']);
        end
        
        if ~exist([params.modelpath featureName,'/kfold', int2str(kfoldIter)], 'dir');
            mkdir([params.modelpath featureName,'/kfold', int2str(kfoldIter)]);
        end
              
        X = []; Y = [];
        
        % Get Train Features
        for i = 1:numel(FeatureData{feat})

            dirs = FeatureData{feat}(i);
            className = {dirs([dirs.isdir]).name};
            className = setdiff(className, {'.','..'});
            if isempty(className)

                continue
            end
            
            files = dir([params.trainfeaturepath, featureName, '/kfold', int2str(kfoldIter),'/',char(className),'/*', params.featurefiletype]);
            
            for j = 1 : numel(files)
                
                % Loose spatial information here
                load([params.trainfeaturepath, featureName, '/kfold', int2str(kfoldIter),'/',char(className),'/',files(j).name], 'fvector');
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
        save( [params.modelpath featureName,'/kfold', int2str(kfoldIter), '/model_to_train_data' params.modelfiletype], 'X', 'Y', '-v7.3');
        libsvmwrite([params.modelpath featureName,'/kfold', int2str(kfoldIter), '/', 'model_to_train_data.txt'], Y, sparse(double(X)));
                
        %Call classifier
        if strcmp('LinearSVM', params.classifiertype)
            [labels, data] = libsvmread([params.modelpath featureName,'/kfold', int2str(kfoldIter), '/', 'model_to_train_data.txt']);
            %Run parametric search to find best parameters based on training
            %data
            C = linear_parametric_search(double(labels),double(data));
            % -s 0 => classification, -t 0 => linear SVM, -c 1 => C (slack) = 1
            model = svmtrain(labels, data, ['-s 0 -t 0 -c ' num2str(C)]);            
        elseif strcmp('NonlinearSVM', params.classifiertype)
            [labels, data] = libsvmread([params.modelpath featureName,'/kfold', int2str(kfoldIter), '/', 'model_to_train_data.txt']);
            %Run parametric search to find best parameters based on training
            %data
            [C,G] = nonlinear_parametric_search(double(labels),double(data));
            % -s 0 => classification, -t 2 => RBF kernel, -c 1 => C (slack) = 1
            model = svmtrain(labels, data, ['-s 0 -t 2 -c ' num2str(C) ' -g ' num2str(G)]);            
        end
        
        save( [params.modelpath featureName,'/kfold', int2str(kfoldIter), '/', params.classifiertype 'model', params.modelfiletype] , 'model', '-v7.3');
        ModelData{feat} = model;
    end
end
end
