classdef ObjectRecognitionPipeline
 
   properties 
        params;         
   end    
      
   methods
       
       function obj = ObjectRecognitionPipeline(pars)
           if pars.verbose
            fprintf('Creating Object Recognition Class ...\n')
           end
           obj.params = pars; 
       end              
        
       function SampleData = MakeData(obj, kfoldIter)
           
           % Assuming that if train folder is there then test folder is
           % also there. If not then this will fail
           if exist([obj.params.samplesdatapath,'kfold', int2str(kfoldIter)], 'dir');
               SampleData = dir([obj.params.samplesdatapath, 'kfold', int2str(kfoldIter)]);
               return;
           else
               
               if obj.params.verbose
                   fprintf('Creating random samples of size [%d %d]\n',obj.params.db.imdims(1), obj.params.db.imdims(2))
               end
                              
               imageFiles = dir(fullfile(obj.params.database,obj.params.imagefiletype));                              
               
               for i = 1:numel(imageFiles)
                   
                   fileCount = 0;
                   
                   if ~exist([obj.params.samplesdatapath,'kfold', int2str(kfoldIter),'/' ,imageFiles(i).name(1:end-4)],'dir')
                      mkdir([obj.params.samplesdatapath,'kfold', int2str(kfoldIter), '/', imageFiles(i).name(1:end-4)]); 
                   end
                   
                   im = imread(fullfile(obj.params.database, imageFiles(i).name));
                   numrowsMini = round(0.2*size(im,1));
                   imMini = imresize(im, [numrowsMini, NaN]);
                   for j = 1:obj.params.db.numsamples
                       RoIfilename = sprintf('%03d',fileCount);
                       RoIpathname = [obj.params.samplesdatapath,'kfold', int2str(kfoldIter), '/', imageFiles(i).name(1:end-4), '/',RoIfilename,'.jpg'];                       
                       rowVal = randi(size(imMini,1)-obj.params.db.imdims(1),1,1);
                       colVal = randi(size(imMini,2)-obj.params.db.imdims(2),1,1);
                       imRoI = imMini(rowVal:rowVal+obj.params.db.imdims(1)-1,colVal:colVal+obj.params.db.imdims(2)-1,:);
                       imwrite(imRoI, RoIpathname, 'jpg');
                       fileCount = fileCount + 1;
                   end
               end               
           SampleData = dir([obj.params.samplesdatapath, 'kfold', int2str(kfoldIter)]);   
           end
       end
        
       function [TrainData, TestData] = SplitData(obj, SampleData, kfoldIter)
           
           % Assuming that if train folder is there then test folder is
           % also there. If not then this will fail
           if exist([obj.params.traindatapath,'kfold', int2str(kfoldIter)], 'dir');
               TrainData = dir([obj.params.traindatapath, 'kfold', int2str(kfoldIter)]);
               TestData = dir([obj.params.testdatapath, 'kfold', int2str(kfoldIter)]);
               return;
           else
               
               if obj.params.verbose
                   fprintf('Splitting Data into Train and Test with split ratio %f\n', obj.params.splitratio)
               end
               
               for s = 1:numel(SampleData)
               dirs = SampleData(s);
               className = {dirs([dirs.isdir]).name};
               className = setdiff(className, {'.','..'});               
               if isempty(className) 
                   continue
               end
               
                   if ~exist([obj.params.traindatapath,'kfold', int2str(kfoldIter),'/' , char(className)],'dir')
                      mkdir([obj.params.traindatapath,'kfold', int2str(kfoldIter), '/', char(className)]); 
                   end
               
                   if ~exist([obj.params.testdatapath,'kfold', int2str(kfoldIter),'/' , char(className)],'dir')
                      mkdir([obj.params.testdatapath,'kfold', int2str(kfoldIter), '/', char(className)]); 
                   end
                   
                   imageFiles = dir([obj.params.samplesdatapath,'kfold', int2str(kfoldIter),'/',char(className),'/',obj.params.imagefiletype]);
                   numImageFiles = size(imageFiles,1);
                   numTrainFiles = floor(obj.params.splitratio*numImageFiles);                   
                   indices = randperm(numImageFiles,numImageFiles);
                   
                   for i=1:numTrainFiles                   
                       try 
                        copyfile([obj.params.samplesdatapath,'kfold', int2str(kfoldIter),'/',char(className),'/', imageFiles(indices(i)).name], [obj.params.traindatapath, 'kfold', int2str(kfoldIter), '/', char(className), '/', imageFiles(indices(i)).name], 'f');   
                       catch ME
                           if (isempty (strfind (ME.message, 'Operation not supported')))
                               rethrow(ME);
                           end
                       end
                   end
                   for i = numTrainFiles+1:numImageFiles
                    try
                        copyfile([obj.params.samplesdatapath,'kfold', int2str(kfoldIter),'/',char(className),'/', imageFiles(indices(i)).name], [obj.params.testdatapath, 'kfold', int2str(kfoldIter), '/', char(className), '/', imageFiles(indices(i)).name], 'f'); 
                    catch ME
                           if (isempty (strfind (ME.message, 'Operation not supported')))
                               rethrow(ME);
                           end
                    end
                   end
               end
               
               TrainData = dir([obj.params.traindatapath, 'kfold', int2str(kfoldIter)]);  
               TestData = dir([obj.params.traindatapath, 'kfold', int2str(kfoldIter)]);                                
           end
       end                        
                  
       function FeatureData = ExtractFeatures(obj, Data, kfoldIter, tag)
            
            if obj.params.verbose
              fprintf('Check Features\n')
           end
            FeatureData = ExtractFeatures(Data, obj.params, kfoldIter, tag);
       end              
       
       function ClsrModel = TrainClassifier(obj, FeatureData, kfoldIter)
           if obj.params.verbose
               fprintf('Check for Trained Classifier Models\n');
           end
           
           ClsrModel = TrainClassifier(FeatureData, obj.params, kfoldIter);          
       end
       
       function ResultData = EvaluateTestData(obj, FeatureData, ClassifierModel, kfoldIter)
            if obj.params.verbose
               fprintf('Check for Predictions\n'); 
            end
            
            ResultData = EvaluateTestData(FeatureData, ClassifierModel, obj.params, kfoldIter);                                   
       end
       
   end
   
end
