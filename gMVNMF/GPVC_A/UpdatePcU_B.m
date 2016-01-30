function [Pc_final, U1_final, U2_final, nIter_final, objhistory_final] = UpdatePcU_B(X1, X2, k, W1, W2, options, U1, U2, Pc)
%U, V are probably initilizations, in case they are empty we do random init

% Notation:
% X1 : Data points in View 1
% X2 : Data points in View 2 
% k : Number of hidden factors
% W1 : weight matrix of the affinity graph (view 1)
% W2 : weight matrix of the affinity graph (view 2)
%
% options ... Structure holding all settings
%
% Parts of the code Written by Deng Cai (dengcai AT gmail.com)

%Options contain minIter, maxIter, error allowed, alpha

error = options.error;
maxIter = options.maxIter;
minIter = options.minIter;
rounds = 1;

if ~isempty(maxIter) && maxIter < minIter                   %Sanity checks and default value initialization
    minIter = maxIter;
end

alpha = options.alpha;                                      %Graph Reg weight

[numFeat,nSmp]=size(X1);                                        %Dimensions

if alpha > 0                                                   %Graph regularisation matrix
    W = alpha*(W1+W2);
    DCol = full(sum(W,2));
    D = spdiags(DCol,[0],nSmp,nSmp);
    L = D - W;
else
    L = [];
end

if isempty(U1)                                           %If empty Pc i.e. need for random intializations
    U1 = abs(rand(size(W1,1),k));
    rounds = options.rounds;
end
if isempty(U2)                                           %If empty Pc i.e. need for random intializations
    U2 = abs(rand(size(W2,1),k));
    rounds = options.rounds;
end
if isempty(Pc)                                           %If empty Pc i.e. need for random intializations
    Pc = abs(rand(nSmp,k));
    rounds = options.rounds;
end

tryNo = 0;

while tryNo < rounds
    tryNo = tryNo+1;
    objValue = [ObjectivePc(Pc, X1, X2, U1, U2, L)];
    nIter = 0;
    while(nIter < maxIter)               
        % ===================== update Pc ========================
        M = X1'*U1 + X2'*U2;
        N = U1'*U1 + U2'*U2;
        PcN = Pc*N;
        if alpha > 0
            WPc = W*Pc;
            DPc = D*Pc;            
            M = M + WPc;
            PcN = PcN + DPc;
        end
        Pc = Pc.*(M./max(PcN,1e-10));

        % ===================== update U's ========================
        PPc = Pc'*Pc;
        XP1 = X1*Pc;
        UPP1 = U1*PPc;
        U1 = U1.*(XP1./max(UPP1,1e-10));

        XP2 = X2*Pc;
        UPP2 = U2*PPc;
        U2 = U2.*(XP2./max(UPP2,1e-10));

        newobj = ObjectivePc(Pc, X1, X2, U1, U2, L);
        objValue = [objValue;newobj];

        nIter = nIter + 1;
        if nIter >= minIter                  %Now check if it has converged or not
            if ((abs(objValue(nIter)-objValue(nIter-1))/objValue(nIter) < error) || (objValue(nIter) <= error))
                break;
            end
        end
    end

    if tryNo == 1
        U1_final = U1;
        U2_final = U2;
        Pc_final = Pc;
        nIter_final = nIter;
        objhistory_final = objValue;
    else
       if objValue(end) < objhistory_final(end)
           U1_final = U1;
           U2_final = U2;
           Pc_final = Pc;
           nIter_final = nIter;
           objhistory_final = objValue;
       end
    end
end    
    %[U_final,V_final] = NormalizeUV(U_final, V_final, NormV, Norm);             %Final Normalisation, from Cai et al

%==========================================================================
function [obj] = ObjectivePc(Pc, X1, X2, U1, U2, L)
    val1 = CalculateObj(X1, U1, Pc, L);
    val2 = CalculateObj(X2, U2, Pc, (L.*0));
    obj = val1 + val2;
    
function [obj, dV] = CalculateObj(X, U, V, L, deltaVU, dVordU)
%Returns graph regularised loss with X = UV, L is the weight matrix (multiplied by its parameter)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    nBlock = ceil(mn/MAXARRAY);

    if mn < MAXARRAY                                    %If complete matrix can be computed in one go
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));                      %Frobenius norm
        if deltaVU                                      %By default, we do not consider this    
            if dVordU
                dV = dX'*U + L*V;
            else
                dV = dX*V;
            end
        end
    else                                                %Computing the matrix in parts
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        PatchSize = ceil(nSmp/nBlock);
        for i = 1:nBlock
            if i*PatchSize > nSmp
                smpIdx = (i-1)*PatchSize+1:nSmp;
            else
                smpIdx = (i-1)*PatchSize+1:i*PatchSize;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V;
            end
        end
    end
    if isempty(L)
        obj_Lap = 0;
    else
        obj_Lap = sum(sum((V'*L).*V'));
    end
    obj = obj_NMF+obj_Lap;                              %Remember that L was already multipled with the paramter to Graph Reg
 
