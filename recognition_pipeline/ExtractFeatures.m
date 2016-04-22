function FeatureData = ExtractFeatures(dataStruct, params, kfoldIter, tag)

FeatureData = cell(length(params.featuretype),1);

if strcmp('train', tag)
featurepath = params.trainfeaturepath;
datapath = params.traindatapath; 
elseif strcmp('test', tag)
featurepath = params.testfeaturepath;
datapath = params.testdatapath; 
end

for feat = 1:length(params.featuretype)
    
    featureName = char(params.featuretype{feat});
    
    if exist([featurepath featureName, '/kfold', int2str(kfoldIter)],'dir')
        if params.verbose
            fprintf([featureName ' Feature Exists\n']);
        end
        FeatureData{feat} = dir([featurepath, featureName, '/kfold', int2str(kfoldIter)]);
    else
        if params.verbose
            fprintf(['Extract ' featureName '\n']);
        end
                       
        %Extract features
        for i = 1:numel(dataStruct)
            
            dirs = dataStruct(i);
               className = {dirs([dirs.isdir]).name};
               className = setdiff(className, {'.','..'});               
               if isempty(className) 
                   continue
               end
               
            if ~exist([featurepath, featureName, '/kfold', int2str(kfoldIter),'/' , char(className)],'dir')
                      mkdir([featurepath, featureName, '/kfold', int2str(kfoldIter), '/', char(className)]); 
            end    
                                          
            if strcmp('train', tag)               
                imageFiles = dir([datapath,'kfold', int2str(kfoldIter),'/',char(className),'/',params.imagefiletype]);               
            elseif strcmp('test', tag)
                imageFiles = dir([datapath,'kfold', int2str(kfoldIter),'/',char(className),'/',params.imagefiletype]);                
            end
               
            for j = 1:numel(imageFiles)
                im = imread([datapath 'kfold' int2str(kfoldIter) '/' char(className), '/', imageFiles(j).name]);
            
            if strcmp(featureName, 'gist')
                fvector = feval(params.gist.extractfunction, im, '', params.gist);                
            end                       
                            
            save( [featurepath featureName,'/kfold', int2str(kfoldIter) ,'/',char(className),'/', imageFiles(j).name(1:end-4), params.featurefiletype] , 'fvector');
                                                
            end
        end
                
        FeatureData{feat} = dir([featurepath, featureName, '/kfold', int2str(kfoldIter)]);        
    end
end

end
