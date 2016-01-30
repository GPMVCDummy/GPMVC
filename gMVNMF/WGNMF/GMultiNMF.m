function [finalU, finalV, finalcentroidV, finalweights, log] = GMultiNMF(X, K, W, label,options)
%	Notation:
% 	X ... a cell containing all views for the data
% 	K ... number of hidden factors
% 	W ... weight matrix of the affinity graph 
% 	label ... ground truth labels

%	Writen by Jialu Liu (jliu64@illinois.edu)
% 	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)

%	References:
% 	J. Liu, C.Wang, J. Gao, and J. Han, ��Multi-view clustering via joint nonnegative matrix factorization,�� in Proc. SDM, Austin, Texas, May 2013, vol. 13, pp. 252�C260.
% 	Zhenfan Wang, Xiangwei Kong, Haiyan Fu, Ming Li, Yujia Zhang, FEATURE EXTRACTION VIA MULTI-VIEW NON-NEGATIVE MATRIX FACTORIZATION WITH LOCAL GRAPH REGULARIZATION, ICIP 2015.

% Weights represent the weights for each view (They are common for both the
% divergence with consensus and also the graph laplacian
% options.gamma represents the parameter for handling the weights
% options.beta is or handling the graph regularization term
% options.varWeight is 1 if we want to weights to be varied, otherise it's 0

tic;
viewNum = length(X);
Rounds = options.rounds;
gamma = options.gamma;
beta = options.beta;
delta = options.delta;
nSmp=size(X{1},2);

U = cell(1, viewNum);
V = cell(1, viewNum);
L = cell(1,viewNum);

log = 0;
ac=0;

%% Compute the graph laplacians
for i = 1:viewNum
        Wtemp = options.alpha*beta*W{i};
        DCol = full(sum(Wtemp,2));
        D = spdiags(DCol,0,nSmp,nSmp);
        L{i} = D - Wtemp;
end
%%

%% Initiliaze weights
weights(1:viewNum) = (1/viewNum);
%%

%% initialize basis and coefficient matrices, initialize on the basis of standard GNMF algorithm
tic;
Goptions.maxIter = options.maxIter;
Goptions.alpha=options.alpha*beta;
rand('twister',5489);
[U{1}, V{1}] = GNMF(X{1}, K, W{1}, Goptions);        %In this case, random inits take place
rand('twister',5489);
%printResult(V{1}, label, options.K, options.kmeans);        
for i = 2:viewNum
    rand('twister',5489);
    [U{i}, V{i}] = GNMF(X{i}, K, W{i}, Goptions);
    rand('twister',5489);
    %printResult(V{i}, label, options.K, options.kmeans);
end
toc;
%%

%%
for i = 1:viewNum
    [U{i}, V{i}] = Normalize(U{i}, V{i});
end
%%

optionsForPerViewNMF = options;
oldac=0;
maxac=0;
j = 0;
sumRound=0;
while j < Rounds
    sumRound=sumRound+1;
    j = j + 1;
    
    %Update centroid V
    centroidV = (weights(1)^gamma)*V{1};
    for i = 2:viewNum
        centroidV = centroidV + (weights(i)^gamma)*V{i};
    end
    centroidV = centroidV / sum(weights.^gamma);            %Check if the array is modified or not

    %Compute loss
    logL = 0;
    logy = [];
    for i = 1:viewNum
        alpha = weights(i)^gamma;
        tmp1 = (X{i} - U{i}*V{i}');
        tmp2 = (V{i} - centroidV);
        logL = logL + alpha*(sum(sum(tmp1.^2)) + delta*(sum(sum(tmp2.^2)))+sum(sum((V{i}'*L{i}).*V{i}')));
        logy(i) = logL;
    end
    %fprintf('LOG1 %.9f, ',logL);
    %fprintf('%.10f %.10f %.10f\n',logL,logy(1),logy(2)-logy(1));

    H = [];
    vale = [];
    costs = [];
    sMean = 0;
    for i = 1:viewNum
        sMean = sMean + size(X{i},1);
    end
    sMean = sMean/viewNum;
    for i = 1:viewNum
        tmp1 = (X{i} - U{i}*V{i}');
        tmp2 = (V{i} - centroidV);
        val = gamma*(sum(sum(tmp1.^2))+delta*(sum(sum(tmp2.^2)))+sum(sum((V{i}'*L{i}).*V{i}')));  
        costs(end+1) = (sMean*(sum(sum(tmp1.^2))/size(X{i},1)))+0*delta*(sum(sum(tmp2.^2)))+0*sum(sum((V{i}'*L{i}).*V{i}'));
        vale(end+1) = val;
        val = val^(1.0/(1-gamma));
        H(end+1) = val;
    end

    %Update the weights if the corresponding option is set
    if (options.varWeight > 0)
        %H
        %vale
        tmpSum = sum(H);
        weights = (H./tmpSum);    
    end
    
    for i=1:viewNum
        fprintf('%.3f ',weights(i));
    end
    fprintf(' comp %d\n',j);
    
    for i=1:viewNum
        fprintf('%.9f ',costs(i));
    end
    fprintf(' are the costs %d\n',j);
    
    %Compute loss
    logL = 0;
    logy = [];
    for i = 1:viewNum
        alpha = weights(i)^gamma;
        tmp1 = (X{i} - U{i}*V{i}');
        tmp2 = (V{i} - centroidV);
        logL = logL + alpha*(sum(sum(tmp1.^2)) + delta*(sum(sum(tmp2.^2)))+sum(sum((V{i}'*L{i}).*V{i}')));
        logy(i) = logL;
    end
    %fprintf('LOG1 %.9f, ',logL);
    %fprintf('%.10f %.10f %.10f\n',logL,logy(1),logy(2)-logy(1));
    
    log(end+1)=logL;
    rand('twister',5489);
    ac = ComputeStats(centroidV, label, options.K, 1, 1);
    if ac > maxac
        maxac = ac;
        finalU=U;
        finalV=V;
        finalcentroidV=centroidV;
        finalweights = weights;
    end
    
    if sumRound==Rounds
        break;
    end
    
    %break;    
    
    %weights = finalweights;
    
    %Update the individual view, the weights do not have any role here
    for i = 1:viewNum
        %optionsForPerViewNMF.alpha = options.Gaplpha;
        rand('twister',5489);
        [U{i}, V{i}] = PerViewNMF(X{i}, K, centroidV, W{i}, optionsForPerViewNMF, U{i}, V{i}); 
    end
    
end
toc


function [U, V] = Normalize(U, V)
    [U,V] = NormalizeUV(U, V, 0, 1);

function [U, V] = NormalizeUV(U, V, NormV, Norm)
    nSmp = size(V,1);
    mFea = size(U,1);
    if Norm == 2
        if NormV
            norms = sqrt(sum(V.^2,1));
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sqrt(sum(U.^2,1));
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    else
        if NormV
            norms = sum(abs(V),1);
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sum(abs(U),1);
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = bsxfun(@times, V, norms);
        end
    end
