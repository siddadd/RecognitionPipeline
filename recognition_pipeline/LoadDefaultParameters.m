function params = LoadDefaultParameters
% This function loads default parameters
% 05/17/2016
% Siddharth Advani
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Loading default parameters ... \n')

%% Debug Params
params.verbose     = 'true'            ; 
params.run = 'full'; % 'fast';
%% Feature Params
params.featuretype = {'gist'}          ;
params.gist.extractfunction = @LMgist  ; 
params.gist.imageSize = [256 256]; % it works also with non-square images
params.gist.orientationsPerScale = [8 8 8 8];
params.gist.numberBlocks = 4;
params.gist.fc_prefilt = 4;
%% Classifier Params
%params.classifiertype = 'LinearSVM'    ;
params.classifiertype = 'NonlinearSVM' ;
params.splitratio     = 0.5            ;
params.kfold          = 1              ;
%% File Params
params.traindatapath  = '../cache/traindata/';
params.testdatapath  = '../cache/testdata/'   ;
params.samplesdatapath = '../cache/sampledata/';
params.trainfeaturepath = '../cache/trainfeaturedata/' ;
params.testfeaturepath = '../cache/testfeaturedata/' ;
params.modelpath = '../cache/modeldata/'     ;
params.resultpath = '../cache/resultdata/'   ;
params.gtfiletype     = ''                   ;
params.imagefiletype  = '*.jpg'              ;
params.featurefiletype = '.mat'              ;
params.modelfiletype = '.mat'                ;
params.resultfiletype = '.mat'               ;
%% Database Params
params.database       = './Datasets/Wegmans/Images/';
params.db.imdims      = [256, 256]        ;
params.db.numsamples  = 20; % take these samples from every image
params.classnames     = {'Cereal', 'Chips', 'Cleaning', 'Coffee', 'Condiments', 'Cookies', 'Dental', 'Juice', 'Pasta', 'Refrigerated', 'Sauce', 'Soda', 'Soup', 'Storage'};
params.validclasses   = [1 1 1 1 1 1 1 1 1 1 1 1 1 1];

end
