function Experiment_Indian_SSWK_MEDA() %#ok<*FNDEF>
% This shows how to use SSWK-MEDA in a 1-nearest neighbor classifier in indian_pines dataset.
  dbstop if error;
  data = 'Indian_pines';
  addpath(genpath('E:\梁天炀迁移学习资料整理'));
  load('data\Indian_pines\Indian_pines_corrected.mat');     % source domain
  load('data\Indian_pines\Indian_pines_gt.mat');    
  X = data;
  Y = groundT;
%% classes reshape
  Y(find(Y == 1)) = 0;Y(find(Y == 6)) = 0;Y(find(Y == 7)) = 0;  %#ok<*FNDSB>
  Y(find(Y == 8)) = 0;Y(find(Y == 9)) = 0;Y(find(Y == 10)) = 0;
  Y(find(Y == 13)) = 0;Y(find(Y == 14)) = 0;Y(find(Y == 16)) = 0;
  
  Y(find(Y == 2)) = 1; Y(find(Y == 3)) = 2; Y(find(Y == 4)) = 3;
  Y(find(Y == 5)) = 4;  Y(find(Y == 12)) = 5; Y(find(Y == 15)) = 6;Y(find(Y == 11)) = 7;
 %% choose transfer task
 task = 1;   %%task = 0 or 1
 if task == 0
     Xs = X;Ys = Y;        %%choose big transfer to small
     Ys(8:82,11:42,:) = 0;
     Xt = X(8:82,11:42,:);
     Yt = Y(8:82,11:42,:);
     fprintf('  SSWK_MEDA for Indian Pines: Big transfer to Small \n');
 elseif task == 1
     Xs = X(8:82,11:42,:);  %%choose small transfer to big
     Ys = Y(8:82,11:42,:);
     Xt = X;Yt = Y;
     Yt(8:82,11:42,:) = 0;
     fprintf('  SSWK_MEDA for Indian Pines: Small transfer to Big \n');
 end
 
%SSWK_MEDA    
    options = [];options.T = 5;options.kernel_type = 'rbf';options.winsize = 1;
    if task == 0                                      %big2small   
        options.lambda = 4;  options.eta = 0.0001; 
        options.rho = 0.1; options.dim = 40;  
    elseif task == 1                                  %small2big   
        options.lambda = 0.01;options.eta = 0.0001;
        options.rho = 0.001;options.dim = 70;   
    end
    [acc_SSWK_MEDA, acc_iter_SSWK_MEDA, Beta_SSWK_MEDA, Yt, Yt_SSWK_MEDA] = SSWK_MEDA(Xs, Ys, Xt, Yt, options);
    [w_SSWK_MEDA, OA_SSWK_MEDA, KAPPA_SSWK_MEDA] = pingding(Yt, Yt_SSWK_MEDA);
    fprintf('  SSWK_MEDA OA %1.2f\n', OA_SSWK_MEDA*100);
    
    a=1;
