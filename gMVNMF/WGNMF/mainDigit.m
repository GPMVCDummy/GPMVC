clear;                      %Remove all variables from the workspace
%clc;
 
addpath(genpath('../../partialMV/PVC/recreateResults/measure/'));
addpath(genpath('../../partialMV/PVC/recreateResults/misc/'));
addpath(genpath(('../code/')));
addpath('../tools/');
addpath('../print/');
addpath('../');

options = [];
options.maxIter = 100;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 30;
options.WeightMode='Binary';
options.varWeight = 0;
options.kmeans = 1;

options.Gaplpha=0;                            %Graph regularisation parameter
options.alpha=0;
options.delta = 0.1;
options.beta = 0;
options.gamma = 2;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
dataname={'orl'};
num_views = 2;
numClust = 40;
options.K = numClust;

for idata=1:length(dataname)  
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');        %Just the datafile name
    datafname=cell2mat(dataf(1));       
    load (datafname);                                           %Loading the datafile

    %% normalize data matrix
        X1 = X1 / sum(sum(X1));
        X2 = X2 / sum(sum(X2));
    %%
    Xf1 = X1;                                                     %Directly loading the matrices
    Xf2 = X2;
    X{1} = Xf1;                                                  %View 1
    X{2} = Xf2;                                                  %View 2    
    %X should be row major i.e. rows are the data points
    
   load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
   folds = folds(:,:);
   
   [numFold,numInst]=size(folds);                                   %numInst : numInstances
   
   for f=1:1%numFold
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances

        %Construct manifold
        for i = 1:length(X)
            data{i} = (X{i}(instanceIdx,:))';              %Column major
            options.WeightMode='Binary';
            W{i} = constructW_cai(data{i}',options);           %Need row major
        end
        
        [U, V, centroidV, weights, log] = GMultiNMF(data, options.K, W, truthF, options);
        ComputeStats(centroidV, truthF, options.K);
        fprintf('\n');
    end
end

