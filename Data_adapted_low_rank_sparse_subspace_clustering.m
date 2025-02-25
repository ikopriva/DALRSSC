%% Data_adapted_low_rank_sparse_subspace_clustering
%
% I. Kopriva 2025-02
%
% If using this software please cite the following paper: I. Kopriva,
% "Data-Adaptive Low-Rank Sparse Subspace Clustering,"
% http://arXiv.org/abs/2502.10106
%

%% Initialization
clear
close all

% Set path to all subfolders
addpath(genpath('.'));

%% Load the data from the chosen dataset

% Please uncomment the dataset that you want to use and comment the other ones
% dataName = 'MNIST';
 dataName = 'ORL';
% dataName = 'COIL20'; 

% Number of random partitions
nIter = 20;

% File name where results will be stored
filename = strcat(dataName,'_DALRSSC_Results');

%% STEP 1: prepare data
[paras_data] = params_data(dataName);

i1 = paras_data.i1; i2 = paras_data.i2; % image size
dimSubspace = paras_data.dimSubspace; % Subspace dimension
numIn = paras_data.numIn; % number of in-sample data
numOut = paras_data.numOut; % number of out-of-sample data
nc = paras_data.nc; % number of groups
Y = paras_data.X;  % data
labels = paras_data.labels; % labels

ACC_x_in     = zeros(1, nIter);     ACC_x_out    = zeros(1, nIter);
NMI_x_in     = zeros(1, nIter);     NMI_x_out    = zeros(1, nIter);
Fscore_x_in  = zeros(1, nIter);     Fscore_x_out = zeros(1, nIter);
Rand_x_in    = zeros(1, nIter);     Rand_x_out   = zeros(1, nIter);
Purity_x_in  = zeros(1, nIter);     Purity_x_out = zeros(1, nIter);

affinity_x   = zeros(1, nIter);     

%% STEP 2: Validate data adapted low-rank sparse subspace clustering algorithm together with S0L0 LRSSC algorithm and GMC LRSSC alg
if strcmp(dataName,'COIL20')
   lambda_ipd_star = 0.0; 
   config_ipd_star = 3;
   mu0=3;
elseif strcmp(dataName,'ORL')
   lambda_ipd_star = 0.10; 
   config_ipd_star = 2;
   mu0=2.5;
elseif strcmp(dataName,'MNIST')
   lambda_ipd_star = 0.2;
   config_ipd_star = 1;
   mu0=2;   
end

for it = 1:nIter
    fprintf('Iter: %d\n',it);

    %% Generate a problem instancC
    % Problem instance is a random split of the chosen dataset into an input set (X_in) and an output set (X_out),
    % as well 
    
    %% Prepare in-sample and out-of-sample random partitions
    
    rng('shuffle');
    
    % Each category is separately split, to ensure proportional representation
    nIn = 1; nOut = 1;
    for c=1:nc % Through all categories
        ind = (labels == c); % Indices of the chosen category
        Xc = Y(:,ind);       % Samples ...as the concommitant label sets (label_in, label_out)
        numSamples = size(Xc, 2); % Number of samples ...
        ind = randperm(numSamples); % Random permutation of the indices
        X_in(:,    nIn:nIn+numIn-1 ) = Xc(:, ind(1:numIn)); % Data
        X_out(:, nOut:nOut+numOut-1) = Xc(:, ind(numIn+1:numIn+numOut));
        labels_in(  nIn:nIn + numIn-1) = c; % Labels
        labels_out(nOut:nOut+numOut-1) = c;
        nIn  = nIn  + numIn; % Next indices
        nOut = nOut + numOut;
    end
    X_in( :,   nIn:end) = []; % Cut out the surplus of the allocated space
    X_out(:,  nOut:end) = [];
    labels_in(  nIn:end) = [];
    labels_out(nOut:end) = [];

    %% Validate data adapted low-rank sparse subspace clustering algorithm on the original data X_in
    options = struct('lambda',lambda_ipd_star, 'delta_n_index',config_ipd_star, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',mu0,'max_mu',1e6);
    tstart=tic;
    [Z_x, error] = ADMM_data_adapted_LRSSC_prox_avg(normc(X_in),options);
    cpu_data_adaptive(it) = toc(tstart);
    % IPD
    C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);
    labels_est_data_adapted_LRSSC = SpectralClusteringL(C_sym,nc);
    %labels_est_data_adapted_LRSSC = SpectralClusteringL(abs(Z_x)+abs(Z_x'),nc);

    % estimate basis
    [affinity, B_x_data_adapted, begB_x_data_adapted, enddB_x_data_adapted, mu_X_data_adapted]  = ...
         average_affinity(X_in,labels_est_data_adapted_LRSSC,dimSubspace); 

    if strcmp(dataName,'COIL20')
        options = struct('lambda',0.48, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',4,'max_mu',1e6,'p',0);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S0L0(it)=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S0L0_LRSSC = SpectralClusteringL(C_sym,nc);
        % estimate basis
        [affinity, B_x_S0L0_LRSSC, begB_x_S0L0_LRSSC, enddB_x_S0L0_LRSSC, mu_X_S0L0_LRSSC]  = ...
         average_affinity(X_in,labels_est_S0L0_LRSSC,dimSubspace);

        options = struct('lambda',0.06, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',3,'max_mu',1e6,'p',1/2);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S1o2L1o2=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S1o2L1o2_LRSSC = SpectralClusteringL(C_sym,nc);  
        % estimate basis
        [affinity, B_x_S1o2L1o2_LRSSC, begB_x_S1o2L1o2_LRSSC, enddB_x_S1o2L1o2_LRSSC, mu_X_S1o2L1o2_LRSSC]  = ...
         average_affinity(X_in,labels_est_S1o2L1o2_LRSSC,dimSubspace);

        options = struct('lambda',0.34, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',2,'max_mu',1e6,'p',2/3);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S2o3L203=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S2o3L2o3_LRSSC = SpectralClusteringL(C_sym,nc); 
        % estimate basis
        [affinity, B_x_S2o3L2o3_LRSSC, begB_x_S2o3L2o3_LRSSC, enddB_x_S2o3L2o3_LRSSC, mu_X_S2o3L2o3_LRSSC]  = ...
         average_affinity(X_in,labels_est_S2o3L2o3_LRSSC,dimSubspace);

        options = struct('lambda',0.43, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',3,'max_mu',1e6,'p',1);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S1L1 = toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S1L1_LRSSC = SpectralClusteringL(C_sym,nc); 
        % estimate basis
        [affinity, B_x_S1L1_LRSSC, begB_x_S1L1_LRSSC, enddB_x_S1L1_LRSSC, mu_X_S1L1_LRSSC]  = ...
         average_affinity(X_in,labels_est_S1L1_LRSSC,dimSubspace);
    elseif strcmp(dataName,'MNIST')
        options = struct('lambda',0.48, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',5,'max_mu',1e6,'p',0);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S0L0(it)=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S0L0_LRSSC = SpectralClusteringL(C_sym,nc);
        % estimate basis
        [affinity, B_x_S0L0_LRSSC, begB_x_S0L0_LRSSC, enddB_x_S0L0_LRSSC, mu_X_S0L0_LRSSC]  = ...
         average_affinity(X_in,labels_est_S0L0_LRSSC,dimSubspace);

        options = struct('lambda',0.41, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',4,'max_mu',1e6,'p',1/2);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S1o2L1o2=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S1o2L1o2_LRSSC = SpectralClusteringL(C_sym,nc);  
        % estimate basis
        [affinity, B_x_S1o2L1o2_LRSSC, begB_x_S1o2L1o2_LRSSC, enddB_x_S1o2L1o2_LRSSC, mu_X_S1o2L1o2_LRSSC]  = ...
         average_affinity(X_in,labels_est_S1o2L1o2_LRSSC,dimSubspace);

        options = struct('lambda',0.24, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',4,'max_mu',1e6,'p',2/3);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S2o3L203=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S2o3L2o3_LRSSC = SpectralClusteringL(C_sym,nc); 
        % estimate basis
        [affinity, B_x_S2o3L2o3_LRSSC, begB_x_S2o3L2o3_LRSSC, enddB_x_S2o3L2o3_LRSSC, mu_X_S2o3L2o3_LRSSC]  = ...
         average_affinity(X_in,labels_est_S2o3L2o3_LRSSC,dimSubspace);

        options = struct('lambda',0.31, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',2,'max_mu',1e6,'p',1);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S1L1 = toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S1L1_LRSSC = SpectralClusteringL(C_sym,nc); 
        % estimate basis
        [affinity, B_x_S1L1_LRSSC, begB_x_S1L1_LRSSC, enddB_x_S1L1_LRSSC, mu_X_S1L1_LRSSC]  = ...
         average_affinity(X_in,labels_est_S1L1_LRSSC,dimSubspace);
    elseif strcmp(dataName,'ORL')
        options = struct('lambda',0.01, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',1,'max_mu',1e6,'p',0);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S0L0(it)=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S0L0_LRSSC = SpectralClusteringL(C_sym,nc);
        % estimate basis
        [affinity, B_x_S0L0_LRSSC, begB_x_S0L0_LRSSC, enddB_x_S0L0_LRSSC, mu_X_S0L0_LRSSC]  = ...
         average_affinity(X_in,labels_est_S0L0_LRSSC,dimSubspace);

        options = struct('lambda',0.0, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',2,'max_mu',1e6,'p',1/2);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S1o2L1o2=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S1o2L1o2_LRSSC = SpectralClusteringL(C_sym,nc);  
        % estimate basis
        [affinity, B_x_S1o2L1o2_LRSSC, begB_x_S1o2L1o2_LRSSC, enddB_x_S1o2L1o2_LRSSC, mu_X_S1o2L1o2_LRSSC]  = ...
         average_affinity(X_in,labels_est_S1o2L1o2_LRSSC,dimSubspace);

        options = struct('lambda',0.07, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',2,'max_mu',1e6,'p',2/3);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S2o3L203=toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S2o3L2o3_LRSSC = SpectralClusteringL(C_sym,nc); 
        % estimate basis
        [affinity, B_x_S2o3L2o3_LRSSC, begB_x_S2o3L2o3_LRSSC, enddB_x_S2o3L2o3_LRSSC, mu_X_S2o3L2o3_LRSSC]  = ...
         average_affinity(X_in,labels_est_S2o3L2o3_LRSSC,dimSubspace);

        options = struct('lambda',0.03, 'err_thr',1e-4,'iter_max',100,'rho',3,'mu',2,'max_mu',1e6,'p',1);
        tstart=tic;
        [Z_x, error] = ADMM_LpSp_LRSSC_prox_avg(normc(X_in),options);
        cpu_S1L1 = toc(tstart);
        C_sym = BuildAdjacency_cut(abs(Z_x),dimSubspace);  % IPD
        labels_est_S1L1_LRSSC = SpectralClusteringL(C_sym,nc); 
        % estimate basis
        [affinity, B_x_S1L1_LRSSC, begB_x_S1L1_LRSSC, enddB_x_S1L1_LRSSC, mu_X_S1L1_LRSSC]  = ...
         average_affinity(X_in,labels_est_S1L1_LRSSC,dimSubspace);
    end

    % Performance on in-sample data
    ACC_x_in_data_adapted(it)  = 1 - computeCE(labels_est_data_adapted_LRSSC,labels_in);     
    ACC_x_in_S0L0_LRSSC(it)  = 1 - computeCE(labels_est_S0L0_LRSSC,labels_in);
    ACC_x_in_S1o2L1o2_LRSSC(it)  = 1 - computeCE(labels_est_S1o2L1o2_LRSSC,labels_in);
    ACC_x_in_S2o3L2o3_LRSSC(it)  = 1 - computeCE(labels_est_S2o3L2o3_LRSSC,labels_in);
    ACC_x_in_S1L1_LRSSC(it)  = 1 - computeCE(labels_est_S1L1_LRSSC,labels_in);   
    NMI_x_in_data_adapted(it)  = compute_nmi(labels_est_data_adapted_LRSSC,labels_in);     
    NMI_x_in_S0L0_LRSSC(it)  = compute_nmi(labels_est_S0L0_LRSSC,labels_in);
    NMI_x_in_S1o2L1o2_LRSSC(it)  = compute_nmi(labels_est_S1o2L1o2_LRSSC,labels_in);
    NMI_x_in_S2o3L2o3_LRSSC(it)  = compute_nmi(labels_est_S2o3L2o3_LRSSC,labels_in);
    NMI_x_in_S1L1_LRSSC(it)  = compute_nmi(labels_est_S1L1_LRSSC,labels_in);
    Fscore_x_in_data_adapted(it)  = compute_f(labels_est_data_adapted_LRSSC',labels_in);     
    Fscore_x_in_S0L0_LRSSC(it)  = compute_f(labels_est_S0L0_LRSSC',labels_in);
    Fscore_x_in_S1o2L1o2_LRSSC(it)  = compute_f(labels_est_S1o2L1o2_LRSSC',labels_in);
    Fscore_x_in_S2o3L2o3_LRSSC(it)  = compute_nmi(labels_est_S2o3L2o3_LRSSC,labels_in);
    Fscore_x_in_S1L1_LRSSC(it)  = compute_nmi(labels_est_S1L1_LRSSC,labels_in);

    display('Mean accuracy: Data adapted, S0L0, S1o2L1o2, S2o3L2o3, S1L1:')
    mean(ACC_x_in_data_adapted)
    mean(ACC_x_in_S0L0_LRSSC)
    mean(ACC_x_in_S1o2L1o2_LRSSC)
    mean(ACC_x_in_S2o3L2o3_LRSSC)
    mean(ACC_x_in_S1L1_LRSSC)

    % Clustering of out-of-sample data
    A0=labels_out;  N_out = size(X_out,2);
    X_out = normc(X_out);
    for l=1:nc
        X_outm = X_out - mu_X_data_adapted(:,l);    % make data zero mean for distance calculation
        BB=B_x_data_adapted(:,begB_x_data_adapted(l):enddB_x_data_adapted(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x_data_adapted] = min(D);
    clear D
    % Performance on out-of-sample data with algorithm estimated labels
    ACC_x_out_data_adapted(it)  = 1 - computeCE(A_x_data_adapted,A0); 
    NMI_x_out_data_adapted(it) = compute_nmi(A0,A_x_data_adapted);
    Fscore_x_out_data_adapted(it) = compute_f(A0,A_x_data_adapted);
    clear A_x_data_adapted

   % Clustering of out-of-sample data
    for l=1:nc
        X_outm = X_out - mu_X_S0L0_LRSSC(:,l);    % make data zero mean for distance calculation
        BB=B_x_S0L0_LRSSC(:,begB_x_S0L0_LRSSC(l):enddB_x_S0L0_LRSSC(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x_S0L0_LRSSC] = min(D);
    clear D  

    % Performance on out-of-sample data with algorithm estiated labels
    ACC_x_out_S0L0_LRSSC(it)  = 1 - computeCE(A_x_S0L0_LRSSC,A0); 
    NMI_x_out_S0L0_LRSSC(it) = compute_nmi(A0,A_x_S0L0_LRSSC);
    Fscore_x_out_S0L0_LRSSC(it) = compute_f(A0,A_x_S0L0_LRSSC);
    clear A_x_S0L0_LRSSC

    % Clustering of out-of-sample data
    for l=1:nc
        X_outm = X_out - mu_X_S1o2L1o2_LRSSC(:,l);    % make data zero mean for distance calculation
        BB=B_x_S1o2L1o2_LRSSC(:,begB_x_S1o2L1o2_LRSSC(l):enddB_x_S1o2L1o2_LRSSC(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x_S1o2L1o2_LRSSC] = min(D);
    clear D      

    % Performance on out-of-sample data with algorithm estiated labels
    ACC_x_out_S1o2L1o2_LRSSC(it)  = 1 - computeCE(A_x_S1o2L1o2_LRSSC,A0); 
    NMI_x_out_S1o2L1o2_LRSSC(it) = compute_nmi(A0,A_x_S1o2L1o2_LRSSC);
    Fscore_x_out_S1o2L1o2_LRSSC(it) = compute_f(A0,A_x_S1o2L1o2_LRSSC);
   clear A_x_S1o2L1o2_LRSSC

    % Clustering of out-of-sample data
    for l=1:nc
        X_outm = X_out - mu_X_S2o3L2o3_LRSSC(:,l);    % make data zero mean for distance calculation
        BB=B_x_S2o3L2o3_LRSSC(:,begB_x_S2o3L2o3_LRSSC(l):enddB_x_S2o3L2o3_LRSSC(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x_S2o3L2o3_LRSSC] = min(D);
    clear D      

    % Performance on out-of-sample data with algorithm estiated labels
    ACC_x_out_S2o3L2o3_LRSSC(it)  = 1 - computeCE(A_x_S2o3L2o3_LRSSC,A0); 
    NMI_x_out_S2o3L2o3_LRSSC(it) = compute_nmi(A0,A_x_S2o3L2o3_LRSSC);
    Fscore_x_out_S2o3L2o3_LRSSC(it) = compute_f(A0,A_x_S2o3L2o3_LRSSC);
   clear A_x_S2o3L2o3_LRSSC

   % Clustering of out-of-sample data
    for l=1:nc
        X_outm = X_out - mu_X_S1L1_LRSSC(:,l);    % make data zero mean for distance calculation
        BB=B_x_S1L1_LRSSC(:,begB_x_S1L1_LRSSC(l):enddB_x_S1L1_LRSSC(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x_S1L1_LRSSC] = min(D);
    clear D      

    % Performance on out-of-sample data with algorithm estiated labels
    ACC_x_out_S1L1_LRSSC(it)  = 1 - computeCE(A_x_S1L1_LRSSC,A0); 
    NMI_x_out_S1L1_LRSSC(it) = compute_nmi(A0,A_x_S1L1_LRSSC);
    Fscore_x_out_S1L1_LRSSC(it) = compute_f(A0,A_x_S1L1_LRSSC);
   clear A_x_S1L1_LRSSC

    save (filename,'ACC_x_in_data_adapted', 'NMI_x_in_data_adapted', 'Fscore_x_in_data_adapted',...
        'ACC_x_out_data_adapted', 'NMI_x_out_data_adapted', 'Fscore_x_out_data_adapted',...
        'ACC_x_in_S0L0_LRSSC', 'NMI_x_in_S0L0_LRSSC', 'Fscore_x_in_S0L0_LRSSC',...
        'ACC_x_out_S0L0_LRSSC', 'NMI_x_out_S0L0_LRSSC', 'Fscore_x_out_S0L0_LRSSC',...
        'ACC_x_in_S1o2L1o2_LRSSC', 'NMI_x_in_S1o2L1o2_LRSSC', 'Fscore_x_in_S1o2L1o2_LRSSC',...
        'ACC_x_out_S1o2L1o2_LRSSC', 'NMI_x_out_S1o2L1o2_LRSSC', 'Fscore_x_out_S1o2L1o2_LRSSC',...   
        'ACC_x_in_S2o3L2o3_LRSSC', 'NMI_x_in_S2o3L2o3_LRSSC', 'Fscore_x_in_S2o3L2o3_LRSSC',...
        'ACC_x_out_S2o3L2o3_LRSSC', 'NMI_x_out_S2o3L2o3_LRSSC', 'Fscore_x_out_S2o3L2o3_LRSSC',...
        'ACC_x_in_S1L1_LRSSC', 'NMI_x_in_S1L1_LRSSC', 'Fscore_x_in_S1L1_LRSSC',...
        'ACC_x_out_S1L1_LRSSC', 'NMI_x_out_S1L1_LRSSC', 'Fscore_x_out_S1L1_LRSSC',...
        'cpu_data_adaptive', 'cpu_S1L1', 'cpu_S2o3L203', 'cpu_S1o2L1o2', 'cpu_S0L0');        
end

display('Estimated performances:')
display(' ************* DATA ADAPTED LRSSC ************************')
display('*********** In-sample data:')
mean_ACC_x_in_data_adapted=mean(ACC_x_in_data_adapted)
std_ACC_x_in_data_adapted=std(ACC_x_in_data_adapted)
  
mean_NMI_x_in_data_adapted=mean(NMI_x_in_data_adapted)
std_NMI_x_in_data_adapted=std(NMI_x_in_data_adapted)

mean_Fscore_x_in_data_adapted=mean(Fscore_x_in_data_adapted)
std_Fscore_x_in_data_adapted=std(Fscore_x_in_data_adapted)

display('*********** Out_of sample data:')
mean_ACC_x_out_data_adapted=mean(ACC_x_out_data_adapted)
std_ACC_x_out_data_adapted=std(ACC_x_out_data_adapted)
  
mean_NMI_x_out_data_adapted=mean(NMI_x_out_data_adapted)
std_NMI_x_out_data_adapted=std(NMI_x_out_data_adapted)

mean_Fscore_x_out_data_adapted=mean(Fscore_x_out_data_adapted)
std_Fscore_x_out_data_adapted=std(Fscore_x_out_data_adapted)

display(' ************* S0L0 LRSSC ************************')
display('*********** In-sample data:')
mean_ACC_x_in_S0L0_LRSSC=mean(ACC_x_in_S0L0_LRSSC)
std_ACC_x_in_S0L0_LRSSC=std(ACC_x_in_S0L0_LRSSC)
[p_acc_s0_in,h_acc_s0_in]=ranksum(ACC_x_in_data_adapted,ACC_x_in_S0L0_LRSSC)

mean_NMI_x_in_S0L0_LRSSC=mean(NMI_x_in_S0L0_LRSSC)
std_NMI_x_in_S0L0_LRSSC=std(NMI_x_in_S0L0_LRSSC)
[p_nmi_s0_in,h_nmi_s0_in]=ranksum(NMI_x_in_data_adapted,NMI_x_in_S0L0_LRSSC)

mean_Fscore_x_in_S0L0_LRSSC=mean(Fscore_x_in_S0L0_LRSSC)
std_Fscore_x_in_S0L0_LRSSC=std(Fscore_x_in_S0L0_LRSSC)
[p_F1_s0_in,h_F1_s0_in]=ranksum(Fscore_x_in_data_adapted,Fscore_x_in_S0L0_LRSSC)

display('*********** Out_of sample data:')
mean_ACC_x_out_S0L0_LRSSC=mean(ACC_x_out_S0L0_LRSSC)
std_ACC_x_out_S0L0_LRSSC=std(ACC_x_out_S0L0_LRSSC)
[p_acc_s0_out,h_acc_s0_out]=ranksum(ACC_x_out_data_adapted,ACC_x_out_S0L0_LRSSC)
  
mean_NMI_x_out_S0L0_LRSSC=mean(NMI_x_out_S0L0_LRSSC)
std_NMI_x_out_S0L0_LRSSC=std(NMI_x_out_S0L0_LRSSC)
[p_nmi_s0_out,h_nmi_s0_out]=ranksum(NMI_x_out_data_adapted,NMI_x_out_S0L0_LRSSC)

mean_Fscore_x_out_S0L0_LRSSC=mean(Fscore_x_out_S0L0_LRSSC)
std_Fscore_x_out_S0L0_LRSSC=std(Fscore_x_out_S0L0_LRSSC)
[p_f1_s0_out,h_f1_s0_out]=ranksum(Fscore_x_out_data_adapted,Fscore_x_out_S0L0_LRSSC)


display(' ************* S1o2L1o2 LRSSC ************************')
display('*********** In-sample data:')
mean_ACC_x_in_S1o2L1o2_LRSSC=mean(ACC_x_in_S1o2L1o2_LRSSC)
std_ACC_x_in_S1o2L1o2_LRSSC=std(ACC_x_in_S1o2L1o2_LRSSC)
[p_acc_s1o2_in,h_acc_s1o2_in] = ranksum(ACC_x_in_data_adapted,ACC_x_in_S1o2L1o2_LRSSC) 

mean_NMI_x_in_S1o2L1o2_LRSSC=mean(NMI_x_in_S1o2L1o2_LRSSC)
std_NMI_x_in_S1o2L1o2_LRSSC=std(NMI_x_in_S1o2L1o2_LRSSC)
[p_nmi_s1o2_in,h_nmi_s1o2_in] = ranksum(NMI_x_in_data_adapted,NMI_x_in_S1o2L1o2_LRSSC) 

mean_Fscore_x_in_S1o2L1o2_LRSSC=mean(Fscore_x_in_S1o2L1o2_LRSSC)
std_Fscore_x_in_S1o2L1o2_LRSSC=std(Fscore_x_in_S1o2L1o2_LRSSC)
[p_f1_s1o2_in,h_fscore_s1o2_in] = ranksum(Fscore_x_in_data_adapted,Fscore_x_in_S1o2L1o2_LRSSC) 

display('*********** Out_of sample data:')
mean_ACC_x_out_S1o2L1o2_LRSSC=mean(ACC_x_out_S1o2L1o2_LRSSC)
std_ACC_x_out_S1o2L1o2_LRSSC=std(ACC_x_out_S1o2L1o2_LRSSC)
[p_acc_s1o2_out,h_acc_s1o2_out] = ranksum(ACC_x_out_data_adapted,ACC_x_out_S1o2L1o2_LRSSC)   

mean_NMI_x_out_S1o2L1o2_LRSSC=mean(NMI_x_out_S1o2L1o2_LRSSC)
std_NMI_x_out_S1o2L1o2_LRSSC=std(NMI_x_out_S1o2L1o2_LRSSC)
[p_nmi_s1o2_out,h_nmi_s1o2_out] = ranksum(NMI_x_out_data_adapted,NMI_x_out_S1o2L1o2_LRSSC) 

mean_Fscore_x_out_S1o2L1o2_LRSSC=mean(Fscore_x_out_S1o2L1o2_LRSSC)
std_Fscore_x_out_S1o2L1o2_LRSSC=std(Fscore_x_out_S1o2L1o2_LRSSC)
[p_f1_s1o2_out,h_f1_s1o2_out] = ranksum(Fscore_x_out_data_adapted,Fscore_x_out_S1o2L1o2_LRSSC) 

display(' ************* S2o3L2o3 LRSSC ************************')
display('*********** In-sample data:')
mean_ACC_x_in_S2o3L2o3_LRSSC=mean(ACC_x_in_S2o3L2o3_LRSSC)
std_ACC_x_in_S2o3L2o3_LRSSC=std(ACC_x_in_S2o3L2o3_LRSSC)
[p_acc_s2o3_in,h_acc_s2o3_in] = ranksum(ACC_x_in_data_adapted,ACC_x_in_S2o3L2o3_LRSSC) 

mean_NMI_x_in_S2o3L2o3_LRSSC=mean(NMI_x_in_S2o3L2o3_LRSSC)
std_NMI_x_in_S2o3L2o3_LRSSC=std(NMI_x_in_S2o3L2o3_LRSSC)
[p_nmi_s2o3_in,h_nmi_s2o3_in] = ranksum(NMI_x_in_data_adapted,NMI_x_in_S2o3L2o3_LRSSC) 

mean_Fscore_x_in_S2o3L2o3_LRSSC=mean(Fscore_x_in_S2o3L2o3_LRSSC)
std_Fscore_x_in_S2o3L2o3_LRSSC=std(Fscore_x_in_S2o3L2o3_LRSSC)
[p_f1_s2o3_in,h_fscore_s2o3_in] = ranksum(Fscore_x_in_data_adapted,Fscore_x_in_S2o3L2o3_LRSSC) 

display('*********** Out_of sample data:')
mean_ACC_x_out_S2o3L2o3_LRSSC=mean(ACC_x_out_S2o3L2o3_LRSSC)
std_ACC_x_out_S2o3L2o3_LRSSC=std(ACC_x_out_S2o3L2o3_LRSSC)
[p_acc_s2o3_out,h_acc_s2o3_out] = ranksum(ACC_x_out_data_adapted,ACC_x_out_S2o3L2o3_LRSSC)   

mean_NMI_x_out_S2o3L2o3_LRSSC=mean(NMI_x_out_S2o3L2o3_LRSSC)
std_NMI_x_out_S2o3L2o3_LRSSC=std(NMI_x_out_S2o3L2o3_LRSSC)
[p_nmi_s2o3_out,h_nmi_s2o3_out] = ranksum(NMI_x_out_data_adapted,NMI_x_out_S2o3L2o3_LRSSC) 

mean_Fscore_x_out_S2o3L2o3_LRSSC=mean(Fscore_x_out_S2o3L2o3_LRSSC)
std_Fscore_x_out_S2o3L2o3_LRSSC=std(Fscore_x_out_S2o3L2o3_LRSSC)
[p_f1_s2o3_out,h_f1_s2o3_out] = ranksum(Fscore_x_out_data_adapted,Fscore_x_out_S2o3L2o3_LRSSC) 

display(' ************* S1L1 LRSSC ************************')
display('*********** In-sample data:')
mean_ACC_x_in_S1L1_LRSSC=mean(ACC_x_in_S1L1_LRSSC)
std_ACC_x_in_S1L1_LRSSC=std(ACC_x_in_S1L1_LRSSC)
[p_acc_s1_in,h_acc_s1_in] = ranksum(ACC_x_in_data_adapted,ACC_x_in_S1L1_LRSSC) 

mean_NMI_x_in_S1L1_LRSSC=mean(NMI_x_in_S1L1_LRSSC)
std_NMI_x_in_S1L1_LRSSC=std(NMI_x_in_S1L1_LRSSC)
[p_nmi_s1_in,h_nmi_s1_in] = ranksum(NMI_x_in_data_adapted,NMI_x_in_S1L1_LRSSC) 

mean_Fscore_x_in_S1L1_LRSSC=mean(Fscore_x_in_S1L1_LRSSC)
std_Fscore_x_in_S1L1_LRSSC=std(Fscore_x_in_S1L1_LRSSC)
[p_f1_s1_in,h_fscore_s1_in] = ranksum(Fscore_x_in_data_adapted,Fscore_x_in_S1L1_LRSSC) 

display('*********** Out_of sample data:')
mean_ACC_x_out_S1L1_LRSSC=mean(ACC_x_out_S1L1_LRSSC)
std_ACC_x_out_S1L1_LRSSC=std(ACC_x_out_S1L1_LRSSC)
[p_acc_s1_out,h_acc_s1_out] = ranksum(ACC_x_out_data_adapted,ACC_x_out_S1L1_LRSSC)   

mean_NMI_x_out_S1L1_LRSSC=mean(NMI_x_out_S1L1_LRSSC)
std_NMI_x_out_S1L1_LRSSC=std(NMI_x_out_S1L1_LRSSC)
[p_nmi_s1_out,h_nmi_s1_out] = ranksum(NMI_x_out_data_adapted,NMI_x_out_S1L1_LRSSC) 

mean_Fscore_x_out_S1L1_LRSSC=mean(Fscore_x_out_S1L1_LRSSC)
std_Fscore_x_out_S1L1_LRSSC=std(Fscore_x_out_S1L1_LRSSC)
[p_f1_s1_out,h_f1_s1_out] = ranksum(Fscore_x_out_data_adapted,Fscore_x_out_S1L1_LRSSC) 



