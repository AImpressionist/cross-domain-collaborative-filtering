% -------------------------------------------------------------------------
%    *** RMGM-OT: Cross-Domain Collaborative Filtering over Time ***
% -------------------------------------------------------------------------
% Written by Bin Li @ University of Technology, Sydney (UTS)
% This code can be used and modified freely for research purpose, please
% cite the following reference when using it:
%    Bin Li et al, "Cross-Domain Collaborative Filtering over Time",
%    IJCAI-11, pp.2293-2298, 2011.
% -------------------------------------------------------------------------
% Inputs:
%    Data: training data, each row is [user_id item_id rating time-slice]
%    Miss: test data, each row is [user_id item_id rating time-slice]
%    K: number of user groups 
%    L: number of item groups
%    ALPHAU: hyperprameters of Dirchlet priors for user-group memberships
%    ALPHAI: hyperprameters of Dirchlet priors for item-group memberships
%    BETA: hyperprameters of Dirchlet priors for rating-scale proportions
%    LAMBDA: precision of Dirchlet priors for user/item-group memberships
%    T_BURNIN; number of Gibbs sampling epochs for burn in
%    T_SAMPLE; number of Gibbs sampling epochs for samples
% Outputs:
%    RMSE: root mean square error on test ratings
% -------------------------------------------------------------------------
function RMSE = rmgmot(Data,Miss,K,L,ALPHAU,ALPHAI,BETA,LAMBDA,T_BURNIN,T_SAMPLE)

N = max([Data(:,1);Miss(:,1)]); % N users
M = max([Data(:,2);Miss(:,2)]); % M items
R = max([Data(:,3);Miss(:,3)]); % R rating scales [1 ... R]
T = max([Data(:,4);Miss(:,4)]); % T time slices [1 ... T]
P = size(Data,1);               % P observed (training) ratings
Q = size(Miss,1);               % Q test ratings

fprintf('- %d users, %d items, %d rating-scales, %d time-slices\n',N,M,R,T);
fprintf('- %d training ratings, %d test ratings\n',P,Q);
fprintf('- training rating-matrix density: %.2f%%\n',100*P/(N*M));

%% Initialize rating counters

ZU = ceil(rand(P,1)*K); % randomly initialize latent user variables
ZI = ceil(rand(P,1)*L); % randomly initialize latent item variables

nTHETAU = zeros(N,K,T); % [user*user-group*time-slice] rating counter
nTHETAI = zeros(M,L,T); % [item*item-group*time-slice] rating counter
nPHI = zeros(K,L,R);    % [user-group*item-group*rating-scale] rating counter

% initialize rating counters
for p = 1:P 
    n = Data(p,1);
    m = Data(p,2);
    r = Data(p,3);
    t = Data(p,4);
    zu = ZU(p);
    zi = ZI(p);
    nTHETAU(n,zu,t) = nTHETAU(n,zu,t)+1;
    nTHETAI(m,zi,t) = nTHETAI(m,zi,t)+1;
    nPHI(zu,zi,r) = nPHI(zu,zi,r)+1;
end

%% Gibbs sampling iterations

THETAU = zeros(N,K,T); % user-group membership matrix
THETAI = zeros(M,L,T); % item-group membership matrix
PHI = zeros(K,L,R); % rating-scale mixing proportions
Fill = zeros(Q,1); % predictions of test ratings

fprintf('- sampling started ...\n');

for iter = 1:(T_BURNIN+T_SAMPLE)
    fprintf('- epoch %d/%d ',iter,T_BURNIN+T_SAMPLE);
    sumBETA = sum(BETA,3);
    sumALPHAU = sum(ALPHAU,2);
    sumALPHAI = sum(ALPHAI,2);
    Order = randperm(P); % shuffle the order of ratings
    
    % sample all the ratings one by one
    for p = 1:P
        i = Order(p);
        n = Data(i,1);
        m = Data(i,2);
        r = Data(i,3);
        t = Data(i,4);
        zu = ZU(i);
        zi = ZI(i);
        nTHETAU(n,zu,t) = nTHETAU(n,zu,t)-1;
        nTHETAI(m,zi,t) = nTHETAI(m,zi,t)-1;
        nPHI(zu,zi,r) = nPHI(zu,zi,r)-1;       
        % compute the conditional probability: eq.(7) in the reference
        ProbR = (nPHI(:,:,r)+BETA(:,:,r))./(sum(nPHI,3)+sumBETA);
        ProbU = nTHETAU(n,:,t)+ALPHAU(n,:,t);
        ProbI = nTHETAI(m,:,t)+ALPHAI(m,:,t);       
        % sample user/item variables from the conditional probability
        UProbs = ProbU'.*ProbR(:,zi);
        zu = sample(UProbs,1);
        IProbs = ProbI.*ProbR(zu,:);
        zi = sample(IProbs,1);  
        % update the counters
        nTHETAU(n,zu,t) = nTHETAU(n,zu,t)+1;
        nTHETAI(m,zi,t) = nTHETAI(m,zi,t)+1;
        nPHI(zu,zi,r) = nPHI(zu,zi,r)+1;
        ZU(i) = zu;
        ZI(i) = zi;
    end
      
    % compute group-level rating matrix
    Core = zeros(K,L);
    sumPHI = sum(nPHI,3);
    for r = 1:R
        PHI(:,:,r) = (nPHI(:,:,r)+BETA(:,:,r))./(sumPHI+sumBETA);
        Core = Core+r*PHI(:,:,r);
    end
    % compute user/item-group memberships
    sumTHETAU = sum(nTHETAU,2);
    sumTHETAI = sum(nTHETAI,2);
    for t = 1:T
        Temp = repmat((sumTHETAU(:,1,t)+sumALPHAU(:,1,t)),[1 K]);
        THETAU(:,:,t) = (nTHETAU(:,:,t)+ALPHAU(:,:,t))./Temp;
        Temp = repmat((sumTHETAI(:,1,t)+sumALPHAI(:,1,t)),[1 L]);
        THETAI(:,:,t) = (nTHETAI(:,:,t)+ALPHAI(:,:,t))./Temp;
    end  

    % predict ratings in the training set
    Fill0 = zeros(P,1);
    for p = 1:P
        n = Data(p,1);
        m = Data(p,2);
        t = Data(p,4);
        Fill0(p) = THETAU(n,:,t)*Core*THETAI(m,:,t)';
    end
    RMSE0 = norm(Data(:,3)-Fill0)/sqrt(P);
    fprintf('[training result on current sample: %f]\n',RMSE0);
    % predict ratings in the test set
    Fill1 = zeros(Q,1);
    for q = 1:Q
        n = Miss(q,1);
        m = Miss(q,2);
        t = Miss(q,4);
        Fill1(q) = THETAU(n,:,t)*Core*THETAI(m,:,t)';
    end
    % average result over samples
    if iter>T_BURNIN
        Fill = (Fill*(iter-T_BURNIN-1)+Fill1)/(iter-T_BURNIN);
    end

    % update user/item-group membership priors
    for t = T:-1:2
        Temp = repmat((sumTHETAU(:,1,t-1)+sumALPHAU(:,1,t-1)),[1 K]);
        ALPHAU(:,:,t) = LAMBDA(1)*(nTHETAU(:,:,t-1)+ALPHAU(:,:,t-1))./Temp;
        Temp = repmat((sumTHETAI(:,1,t-1)+sumALPHAI(:,1,t-1)),[1 L]);
        ALPHAI(:,:,t) = LAMBDA(2)*(nTHETAI(:,:,t-1)+ALPHAI(:,:,t-1))./Temp;
    end
end

%% Output result on the test set

RMSE = norm(Miss(:,3)-Fill)/sqrt(size(Fill,1));
fprintf('- sampling finished [test result over %d samples: %f]\n',T_SAMPLE,RMSE);