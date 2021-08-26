function Experiment_Worldview_SSWK_MEDA()
% This shows how to use SSWK-MEDA algorithms in a 1-nearest neighbor classifier.
dbstop if error;
% data ='houston'; %#ok<*NASGU>
addpath(genpath('E:\梁天炀迁移学习资料整理'));
load('data\Houston\Houston.mat');    
load('data\Houston\Houston_GT.mat');    
load('data\Houston\shadowMask\ShadowMask.mat');    
X=Houston;
X=double(X);
Y=Houston_GT;
[a,b,c]=size(X);
%% classes reshape
Y(find(Y==3))=0;Y(find(Y==4))=0;Y(find(Y==5))=0;Y(find(Y==6))=0; %#ok<*FNDSB>
Y(find(Y==7))=0;Y(find(Y==8))=0;Y(find(Y==9))=0;Y(find(Y==12))=0;
Y(find(Y==14))=0;Y(find(Y==13))=0;Y(find(Y==15))=0;

Y(find(Y==10))=3;Y(find(Y==11))=4;
 %% choose transfer task
shadow = zeros(a,b); bright = zeros(a,b);
for i = 1 : a
    for j = 1 : b
        if ShadowMask(i,j) == 0
            bright(i,j) = Y(i,j);
        else
            shadow(i,j) = Y(i,j);
        end
    end
end
   task = 0;    %%task = 0 or 1
if task == 0  
    Ys = shadow; Yt = bright;
    fprintf('  SSWK_MEDA for Houston: Shadow transfer to Bright \n');
elseif task == 1
    Ys = bright; Yt = shadow;
    fprintf('  SSWK_MEDA for Houston: Bright transfer to Shadow \n');
end

%% run SSWK-MEDA
    options = [];options.T = 5;options.kernel_type = 'rbf';options.winsize = 1;
    if task == 0
        options.lambda = 1;  options.eta = 1;  
        options.rho = 8; options.dim = 40;     %shadow2bright
    elseif  task == 1
         options.lambda = 1;options.eta = 0.1;
        options.rho = 4;options.dim = 65;      %bright2shadow
    end
    [acc_SSWK_MEDA, acc_iter_SSWK_MEDA, Beta_SSWK_MEDA, Yt, Yt_SSWK_MEDA] = SSWK_MEDA(X, Ys, X, Yt, options);
    [w_SSWK_MEDA, OA_SSWK_MEDA, KAPPA_SSWK_MEDA] = pingding(Yt, Yt_SSWK_MEDA);
    fprintf('  SSWK_MEDA OA %1.2f\n', OA_SSWK_MEDA*100);
    
    a=1;  %break point
