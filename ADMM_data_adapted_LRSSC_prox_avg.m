% Non-convex Low Rank Sparse Subbspace Clustering for estimation of the
% matrix of coefficients:
%
% min\|X-XC\|_F^2 + \lambda\sum(f_delta_n(sig(C),a) + \(1-lambda)\f_delta_n(C) 
%
% INPUTS:
%   X: DxN data matrix with N samples and D features
%   lambda: regularization constant related to sparsity induced regualrizer 
%   opts:  Structure value with following fields:
%          opts.lambda:    coefficients for low-rank constraint
%          opts.delta-n_index: index of combination of one of six dublets
%          (delta,n)
%          opts.mu:  penalty parameter for variable C in augmented Lagrangian
%          opts.max_mu:  maximum  penalty parameter for mu parameter
%          opts.rho: step size for adaptively changing mu
%          opts.error_bound: error bound for convergence
%          opts.iter_max:  maximal number of iterations
%
% OUTPUTS:
%   C: NxN matrix of coefficients
%   RMSE: error
%   error: ||X-XC||/||X||
%
% Ivica Kopriva, Januray 2025.


function [C, error] = ADMM_data_adapted_LRSSC_prox_avg(X,opts)

if ~exist('opts', 'var')
    opts = [];
end

% default parameters
rho = 3;
max_mu = 1e6;
err_thr = 1e-4;

if isfield(opts, 'lambda');      lambda = opts.lambda;      end
if isfield(opts, 'mu');      mu = opts.mu;      end  % penalty parameter for ADMM
if isfield(opts,'rho');    rho = opts.rho;  end % rate parameter
if isfield(opts,'max_mu'); max_mu = opts.max_mu; end % maximal value of penalty parameter
if isfield(opts, 'p');  p = opts.p; end  % selection of Lp/Sp norm
if isfield(opts, 'iter_max');    iter_max = opts.iter_max;    end
if isfield(opts, 'err_thr');    err_thr = opts.err_thr;    end
if isfield(opts, 'rho');      rho = opts.rho;      end
if isfield(opts,'delta_n_index'); delta_n_index = opts.delta_n_index; end

%% initialization

[D,N]=size(X);

tau = 1-lambda;

J = zeros(N,N);  % auxiliary variable
C = J;

% Lagrange multpliers
LAM = zeros(N,N);

% Fixed precomputed term for J
tic;

XT = X'*X;

Jf = inv(XT + mu*eye(N));
J = Jf*(XT + mu*C - LAM);
J = normc(J);  % necessary if X is column normalized (YaleB dataset !!!!!!!)

not_converged = 1;
iter=1;

L_grad_vect = [];
err1_all =  [];
err2_all = [];

while not_converged
    
    J_prev = J;
    
    % Update of J
    J = Jf*(XT + mu*C - LAM);
    J = normc(J);  % necessary if X is column normalized (YaleB dataset !!!!!!!)
    
    % Update of C1
    [U Sig V] = svd(J+LAM/mu,'econ');
    
    C1 = U*diag(prox_f_delta_n(diag(Sig)',delta_n_index,lambda/mu))*transpose(V);
    C1 = C1 - diag(diag(C1));  
            
    % Update of C2
    C2 = prox_f_delta_n(J+LAM/mu,delta_n_index,tau/mu);
    C2 = C2 - diag(diag(C2));
    
    % proximal average
    C = lambda*C1 + tau*C2;
    
    % Update of Lagrange multipliers
    LAM = LAM + mu*(J - C);
    
    mu = min(rho*mu, max_mu);
    
    if rho~=1 || iter==1
        Jf = inv(XT + mu*eye(N));
    end
        
    err1 = max(max(abs(J-C)));
    err2 = max(max(abs(J-J_prev)));
    
    % check convergence
    if err1<err_thr && err2<err_thr
        not_converged=0;
        iter
    elseif iter>=iter_max
        not_converged = 0;
        iter;
    else
        iter = iter+1;
    end  
end

error = norm(X-X*J)/norm(X);
end

function [Cth] = prox_f_delta_n(C,index,thr)

sigg = [1/50 1/5 30];
nn = [1.0 1.0 1.5];

sig= sigg(index); n = nn(index); % parameters of f_sig_n surrogate function

u2 = power((n-1)/sig/n,1/n);

tau=thr/sig;

[N1 N2] = size(C);

parfor i1=1:N1   
    x = C(i1,:);
    u2_vec = ones(1,N2)*u2;
    fun = @(u)power(u-abs(x),2)/2 + tau*(1-exp(-sig*power(abs(u),n)));
    u_opt_1=lsqnonlin(fun,u2_vec/2,[],[],optimoptions('lsqnonlin','Display','none')); %,optimoptions('fsolve','Display','iter','Algorithm','trust-region')); %'levenberg-marquardt'));
    u_opt_2=lsqnonlin(fun,abs(x),[],[],optimoptions('lsqnonlin','Display','none'));
    u0 = u_opt_1;
    L_min_1 =  x.*x/2 + u0.*u0/2 - u0.*abs(x) + tau*(1-exp(-sig*power(u0,n)));
    u0 = u_opt_2;
    L_min_2 =  x.*x/2 + u0.*u0/2 - u0.*abs(x) + tau*(1-exp(-sig*power(u0,n)));

    Lmin_1m2 = L_min_1 - L_min_2;
    sign_1 = abs(sign(sign(Lmin_1m2) - 1));
    sign_2 = double(not(sign_1));
    Cth(i1,:) = sign_1.*(sign(x).*u_opt_1);
    Cth(i1,:) = Cth(i1,:) + sign_2.*(sign(x).*u_opt_2); 
end
end

