function [OA,OA_iter,Beta,Yt, Yt_pred] = SSWK_MEDA(Xs_spe_ori,Ys,Xt_spe_ori,Yt,options)
% 本代码是在MEDA的基础上改进的
% Reference:
%% Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S.
%% Yu. Visual Domain Adaptation with Manifold Embedded Distribution
%% Alignment. ACM Multimedia conference 2018.

%% Inputs:
%%% Xs_spe_ori         : Original spectral feature matrix of source domain, n1 * m1 * d
%%% Ys                 : Label matrix of source domain, n1 * m1 * 1
%%% Xt_spe_ori         : Original spectral feature matrix of target domain, n2 * m2 * d
%%% Yt                 : Label matrix of target domain, n2 * m2 * 1
%%% options            : parameter options of SSWK-MEDA:
%%%%% options.d        : dimensions after manifold feature learning (default: 20)
%%%%% options.T        : iteration times (default: 5)
%%%%% options.lambda   : lambda of SSWK-MEDA (default: 0.01)
%%%%% options.eta      : eta of SSWK-MEDA (default: 0.01)
%%%%% options.rho      : rho of SSWK-MEDA (default: 0.01)
%%%%% options.winsize  : window size of spatial filter of SSWK-MEDA (default: 1)
%% Outputs:
%%%% OA                : Final accuracy value
%%%% OA_iter           : Accuracy value list of all iterations, T * 1
%%%% Beta              : Cofficient matrix
%%%  Yt                : reshaped Yt
%%%% Yt_pred           : Prediction labels for target domain

    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.01;
    end
    if ~isfield(options,'lambda')
        options.lambda = 0.01;
    end
    if ~isfield(options,'rho')
        options.rho = 0.01;
    end
    if ~isfield(options,'T')
        options.T = 5;
    end
    if ~isfield(options,'dim')
        options.dim = 20;
    end
    if ~isfield(options,'winsize')
        options.winsize = 1;
    end
    %% Spatial Filtering
    [Xs_spa_ori,Xt_spa_ori]=spatial_filter(Xs_spe_ori,Xt_spe_ori,Ys,Yt,options.winsize);

    [a1,b1,dimen]=size(Xs_spe_ori);[a2,b2,~]=size(Xt_spe_ori);
    Xs_spe_ori = reshape(Xs_spe_ori,a1*b1,dimen);
    Xt_spe_ori = reshape(Xt_spe_ori,a2*b2,dimen);
    Xs_spa_ori = reshape(Xs_spa_ori,a1*b1,dimen);
    Xt_spa_ori = reshape(Xt_spa_ori,a2*b2,dimen);   
    Ys = reshape(Ys,a1*b1,1);
    Yt = reshape(Yt,a2*b2,1);

    index_Xs = find(Ys~=0);      %extract labeled source domain samples
    Ys = Ys(index_Xs,:);
    Xs_spe = Xs_spe_ori(index_Xs,:);
    Xs_spa = Xs_spa_ori(index_Xs,:);
    Xs_spe = scale_normalization(Xs_spe);
    Xs_spa = scale_normalization(Xs_spa);

    index_Xt = find(Yt~=0);      %extract labeled target domain samples
    Yt = Yt(index_Xt,:);
    Xt_spe = Xt_spe_ori(index_Xt,:);
    Xt_spa = Xt_spa_ori(index_Xt,:);
    Xt_spe = scale_normalization(Xt_spe);
    Xt_spa = scale_normalization(Xt_spa);
    
    %% Manifold feature learning
    [Xs_spe_new,Xt_spe_new,~] = GFK_Map(Xs_spe,Xt_spe,options.dim);
    Xs_spe = double(Xs_spe_new');
    Xt_spe = double(Xt_spe_new');
    [Xs_spa_new,Xt_spa_new,~] = GFK_Map(Xs_spa,Xt_spa,options.dim);
    Xs_spa = double(Xs_spa_new');
    Xt_spa = double(Xt_spa_new');
    
    X_spe = [Xs_spe,Xt_spe];
    X_spa = [Xs_spa,Xt_spa];
    n = size(Xs_spe,2);
    m = size(Xt_spe,2);
    C = length(unique(Ys));
    OA_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];

    %% Data normalization
    X_spe = X_spe * diag(sparse(1 ./ sqrt(sum(X_spe.^2))));
    X_spa = X_spa * diag(sparse(1 ./ sqrt(sum(X_spa.^2))));
    
   %% Construct kernel
    K1 = kernel_choice(options.kernel_type,X_spe,sqrt(sum(sum(X_spe .^ 2).^0.5)/(n + m)));
    K2 = kernel_choice(options.kernel_type,X_spa,sqrt(sum(sum(X_spa .^ 2).^0.5)/(n + m)));
    sim=[];
    for i = 1:n+m
       co = corrcoef(X_spe(:,i),X_spa(:,i));
       sim=[sim,co(1,2)];
    end
    afa = mean(sim);
%     afa = 0.9
    K = (1-afa) * K1 +  afa * K2;
    
    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X_spe',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

    % Generate soft labels for the target domain
    knn_model = fitcknn(X_spe(:,1:n)',Ys,'NumNeighbors',1);
    SoftLabel = knn_model.predict(X_spe(:,n + 1:end)');
    
    E = diag(sparse([ones(n,1);zeros(m,1)]));

    for t = 1 : options.T
        % Estimate mu
        mu = estimate_mu(Xs_spe',Ys,Xt_spe',SoftLabel);
        % Construct MMD matrix
        e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
        M = e * e' * length(unique(Ys));
        N = 0;
        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = zeros(n + m,1);
            e(Ys == c) = 1 / length(find(Ys == c));
            e(n + find(SoftLabel == c)) = -1 / length(find(SoftLabel == c));
            e(isinf(e)) = 0;
            N = N + e * e';
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');

        % Compute coefficients vector Beta
        Beta = ((E + options.lambda * M + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E * YY);
        F = K * Beta;
        [~,SoftLabel] = max(F,[],2);

        %% Compute accuracy
        OA = numel(find(SoftLabel(n+1:end)==Yt)) / m;
        SoftLabel = SoftLabel(n+1:end);
        OA_iter = [OA_iter;OA];
    end
    Yt_pred = SoftLabel;
end

function K = kernel_choice(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end