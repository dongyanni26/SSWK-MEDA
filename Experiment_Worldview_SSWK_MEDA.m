function Experiment_Worldview()
% This shows how to use SSWK-MEDA in a 1-nearest neighbor classifier.
 dbstop if error;
 data ='Worldview';
 addpath(genpath('E:\梁天炀迁移学习资料整理'));
 load('data\Worldview2\time_1_sub.mat');     % source domain
 load('data\Worldview2\class_2011.mat');    
 load('data\Worldview2\time_2_sub.mat');     % source domain
 load('data\Worldview2\class_2012.mat');
 
 %% choose transfer task
 task = 0;   %%task = 0 or 1
 if task == 0
     Xss = time_1_sub; Yss = class_2011; Xtt = time_2_sub; Ytt = class_2012;  %%choose 2011 transfer to 2012
     fprintf('  SSWK_MEDA for Worldview: 2011 transfer to 2012 \n');
 elseif task == 1
     Xss = time_2_sub; Yss = class_2012; Xtt = time_1_sub; Ytt = class_2011;  %%choose 2012 transfer to 2011
     fprintf('  SSWK_MEDA for Worldview: 2012 transfer to 2011 \n');
 end
 %% data processing
 Ys = reshape(Yss,400,400,1); Yt = reshape(Ytt,400,400,1);
 Xs = reshape(Xss,400,400,8);Xt = reshape(Xtt,400,400,8);
 Xs = Xs(1:200,201:400,:); Xt = Xt(1:200,201:400,:);
 Ys = Ys(1:200,201:400,:); Yt = Yt(1:200,201:400,:);

%SSWK-MEDA
    options = [];options.T = 5;options.kernel_type = 'rbf';options.winsize = 1;
    if task == 0
        options.lambda = 0.01;options.eta = 8;
        options.rho =  1;options.dim = 4;   %2011to2012   
    elseif task == 1
        options.lambda = 8;options.eta = 1;
        options.rho = 0.01;options.dim = 4;    %2012to2011
    end
    [acc_SSWK_MEDA, acc_iter_SSWK_MEDA, Beta_SSWK_MEDA, Yt, Yt_SSWK_MEDA] = SSWK_MEDA(Xs, Ys, Xt, Yt, options);
    [w_SSWK_MEDA, OA_SSWK_MEDA, KAPPA_SSWK_MEDA] = pingding(Yt, Yt_SSWK_MEDA);
    fprintf('  SSWK_MEDA OA %1.2f\n', OA_SSWK_MEDA*100);
    
    a=1;
