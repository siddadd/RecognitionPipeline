% ***************************************************************************************************************
% AISLE RECOGNITION USING MIT GIST
%
% Author : Siddharth Advani
%
% Date: 05/04/2015
%
% Assumptions : Evaluated on Wegmans Dataset
%
% References :
%             (1) http://people.csail.mit.edu/torralba/code/spatialenvelope/
%                          
% Features : MIT GIST
%
% Dependencies : LibSVM
%
% Comments : Platform tested on is MATLAB R2014a on Linux
% ***************************************************************************************************************

clc; clear all; close all; 

% Add libraries
addpath('./libsvm-3.20/matlab/');
addpath('./libsvm-3.20/tools/');

params = LoadDefaultParameters; 

ORP = ObjectRecognitionPipeline(params); 

for i = 1 : ORP.params.kfold    
    
    fprintf('|-------------------|\n');
    fprintf('Begin kfold %d\n',i);
    
    SampleData = ORP.MakeData(i);
    
    [TrainData, TestData] = ORP.SplitData(SampleData,i);        
    
    TrainFeatureData = ORP.ExtractFeatures(TrainData,i,'train');       

    ClassifierModel = ORP.TrainClassifier(TrainFeatureData, i);
    
    TestFeatureData = ORP.ExtractFeatures(TestData,i,'test');       
    
    ResultData = ORP.EvaluateTestData(TestFeatureData, ClassifierModel, i);
    
end
