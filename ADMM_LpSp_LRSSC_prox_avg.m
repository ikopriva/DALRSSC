% Non-convex Low Rank Sparse Subbspace Clustering for estimation of the
% matrix of coefficients:
% min\|X-XC\|_F^2 + \tau\|C\|_p + \lambda\sum(sig(C)_p)
%
% INPUTS:
%   X: PxN data matrix with n samples and p features
%   lambda: regularization constant related to Lp sparsity induced regualrizer of singular values
%   tau: weight of Lp norm regularization for entries of S
%   affine: 1 - affine subspace model; 0 - independent subspace model
%   opts:  Structure value with following fields:
%          opts.p:    p norm indicator: 1, 2/3, 1/2 or 0.
%          opts.lambda:    coefficients for low-rank constraint
%          opts.mu:  penalty parameter for auxiliary variable J in augmented Lagrangian
%          opts.max_mu:  maximum  penalty parameter for mu1 parameter
%          opts.rho: step size for adaptively changing mu, if 1 fixed mu is used
%          opts.error_bound: error bound for convergence
%          opts.iter_max:  maximal number of iterations
%          opts.d:  subspace dimension used for threholding values of C
%                   matrix
%
% OUTPUTS:
%   C: NxN matrix of coefficients
%   RMSE: error
%   error: ||X-XC||/||X||
%
% Ivica Kopriva, January 2025.

function [C, error] = ADMM_LpSp_LRSSC_prox_avg(X,opts)

if ~exist('opts', 'var')
    opts = [];
end

% default parameters
rho = 3;
max_mu = 1e6;
err_thr = 1e-4;
iter_max = 100;

if isfield(opts, 'lambda');      lambda = opts.lambda;      end
if isfield(opts, 'mu');      mu = opts.mu;      end  % penalty parameter for ADMM
if isfield(opts,'rho');    rho = opts.rho;  end % rate parameter
if isfield(opts,'max_mu'); max_mu = opts.max_mu; end % maximal value of penalty parameter
if isfield(opts, 'p');  p = opts.p; end  % selection of Lp/Sp norm
if isfield(opts, 'iter_max');    iter_max = opts.iter_max;    end
if isfield(opts, 'err_thr');    err_thr = opts.err_thr;    end
if isfield(opts, 'rho');      rho = opts.rho;      end
if isfield(opts,'d');         d = opts.d;   end

%% initialization

[M,N]=size(X);
tau = 1-lambda;

J = zeros(N,N);  % auxiliary variable
C = J;

% Lagrange multpliers
LAM = zeros(N,N);

% Fixed precomputed term for J
XT = X'*X;

Jf = inv(XT + mu*eye(N));
J = Jf*(XT + mu*C - LAM);
J = normc(J);  % necessary if X is column normalized (YaleB dataset !!!!!!!)

not_converged = 1;
iter=1;
err1_all =  [];
err2_all = [];

while not_converged
    
    J_prev = J;
    
    % Update of J
    J = Jf*(XT + mu*C - LAM);
    J = normc(J);  % necessary if X is column normalized (YaleB dataset !!!!!!!)
    
    % Update of Prox_f_C
    %[U,Sig,V]=svdsecon(J+LAM_1/mu1,k);
    [U Sig V] = svd(J+LAM/mu,'econ');
    sig = diag(Sig)';
    thr = lambda/mu;
    
    if p == 1   % S_1 norm
        sig_thr = sign(sig).*max(abs(sig)-thr,0);
    elseif p == 2/3 % S_2/3 norm
        fi_2o3 = (2/3)*power(3*power(thr,3),1/4);
        if thr == 0
            thrm=0.05;
        else
            thrm = thr;
        end
        fi = acosh(27*sig.*sig/16*power(thrm,-1.5));
        c = (2/sqrt(3))*power(thrm,1/4)*sqrt(cosh(fi/3));
        omega = real(power((c + sqrt(2*abs(sig)./c - c.*c))/2,3));
        sig_thr = sign(max(abs(sig)-fi_2o3,0)).*omega;
        sig_thr = max(sig_thr,0);
    elseif p == 1/2 % S_1/2 norm
        psi = acos(thr/8*power(abs(sig)/3,-1.5));
        fi_0p5 = 3/4*power(thr,2/3);
        sig_thr = 2/3*(sig.*(1 + cos(2*pi/3 - 2*psi/3))).*sign(max(abs(sig)-fi_0p5,0));
    elseif p == 0 % S_0 quasi-norm
        thr = sqrt(2*thr);
        sig_thr = sig.*(sign(max(abs(sig)-thr,0)));
    end

    [is inds] = sort(sig_thr,'descend');
    ind = inds(1:sum(sign(is)));
    sig = sig_thr(ind);
    V = V(:,ind);
    U = U(:,ind);
    Sig = diag(sig);
    Prox_f_C = U*Sig*V';
    Prox_f_C = Prox_f_C - diag(diag(Prox_f_C));  
           
    % Update of Prox_g_C
    thr = tau/mu;
    tmp=J + LAM/mu;

    if p == 1   % L_1 norm
        Prox_g_C = sign(tmp).*max(abs(tmp)-thr,0);
    elseif p == 2/3 % L_2/3 norm
        fi_2o3 = (2/3)*power(3*power(thr,3),1/4);
        if thr == 0
            thrm=0.05;
        else
            thrm = thr;
        end
        fi = acosh(27*tmp.*tmp/16*power(thrm,-1.5));
        c = (2/sqrt(3))*power(thrm,1/4)*sqrt(cosh(fi/3));
        omega = power((c + sqrt(2*abs(tmp)./c - c.*c))/2,3);
        Prox_g_C = sign(max(abs(tmp)-fi_2o3,0)).*omega;
    elseif p == 1/2 % L_1/2 norm
        psi = acos(thr/8*power(abs(tmp)/3,-1.5));
        fi_0p5 = 3/4*power(thr,2/3);
        Prox_g_C = 2/3*tmp.*((1 + cos(2*pi/3 - 2*psi/3))).*sign(max(abs(tmp)-fi_0p5,0));
    elseif p == 0 % L_0 quasi-norm
        thr = sqrt(2*thr);
        Prox_g_C = tmp.*(sign(max(abs(tmp)-thr,0)));
    end
    Prox_g_C = Prox_g_C - diag(diag(Prox_g_C));
    
    % proximal average
    C = lambda*Prox_f_C + tau*Prox_g_C;
    
    % Update of Lagrange multipliers
    LAM = LAM + mu*(J - C);
    
    mu = min(rho*mu, max_mu);
    
    if rho~=1 || iter==1
        Jf = inv(XT + mu*eye(N));
    end
        
    err1 = max(max(abs(J-C)));
    err2 = max(max(abs(J-J_prev)));
      
    % err1_all = [err1_all ; err1];
    % err2_all = [err2_all ; err2];
    
    % check convergence
    if err1<err_thr && err2<err_thr
        not_converged=0;
        iter;
    elseif iter>=iter_max
        not_converged = 0;
        iter;
    else
        iter = iter+1;
    end

    iter = iter+1;    
end
%iter

%plot(1:length(err1_all), err1_all, '.-');
%figure;
%plot(1:length(err2_all), err2_all, '.-');
%plot(1:length(L_grad_vect), L_grad_vect, '.-');
%L_grad_vect

%C = normc(C);
%C=C1;

error = norm(X-X*J)/norm(X);
end

