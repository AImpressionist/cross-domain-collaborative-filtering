% load Data.mat
% load Split.mat
%
% "Data" is a P*4 matrix, P is the total number of ratings
% Each row in Data has the form of [user_id item_id rating time-slice]
% For example:
% Data = 
%   [1 2 3 1     : [user_id=1 item_id=2 rating=3 time-slice=1]  
%    2 3 5 2     : [user_id=2 item_id=3 rating=5 time-slice=2]
%    3 1 2 3     : [user_id=3 item_id=1 rating=2 time-slice=3]
%    4 2 4 1     : [user_id=4 item_id=2 rating=4 time-slice=1]
%    ... ...]
% 
% "Split" is a P*1 matrix, P is the total number of ratings. The elements
% of "Split" is in [1 ... S] to indicate split index. 
% Each run of the function "rmgmot" is trained on (S-1) data splits and
% test on the rest one. 

%% fixed parameters below

N = max(Data(:,1));  % number of users
M = max(Data(:,2));  % number of items
R = max(Data(:,3));  % R rating scales [1 ... R]
T = max(Data(:,4));  % T time slices [1 ... T]
S = max(Split(:,1)); % S data splits [1 ... S]

%% customized parameters below

K = 20; %%% number of user groups
L = 20; %%% number of item groups

ALPHAU = ones(N,K,T)/K; %%% hyperprameters of Dirchlet priors for user-group memberships
ALPHAI = ones(M,L,T)/L; %%% hyperprameters of Dirchlet priors for item-group memberships
BETA = ones(K,L,R)/R;   %%% hyperprameters of Dirchlet priors for rating-scale proportions
LAMBDA = ones(1,2)*10;  %%% precision of Dirchlet priors for user/item-group memberships

T_BURNIN = 200; %%% number of Gibbs sampling epochs for burn in
T_SAMPLE = 100; %%% number of Gibbs sampling epochs for samples

%% run experiments over S splits 

RMSE = zeros(S,1); % save overall results of S splits
for s = 1:S
    Train = Data((Split~=s),:);
    Test = Data((Split==s),:);
    fprintf('*** split %d ***\n',s);
    tic
    RMSE(s) = rmgmot(Train,Test,K,L,ALPHAU,ALPHAI,BETA,LAMBDA,T_BURNIN,T_SAMPLE);
    toc
end
fprintf('*** average test result over %d splits: %f ***\n',S,sum(RMSE)/S);

%% visualize user-group membership drifting over time

% n = 1; % select a user 
% imagesc(squeeze(THETAU(n,:,:))); % plot the figure
% colormap gray