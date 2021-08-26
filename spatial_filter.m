function [Xs_spa,Xt_spa]=spatial_filter(Xs_spe,Xt_spe,Ys,Yt,win)
%% Inputs:
%%% Xs_spe     : Spectral feature matrix of source domain, n1 * m1 * d
%%% Xt_spe     : Spectral feature matrix of target domain, n2 * m2 * d
%%% Ys         : Label matrix of source domain, n1 * m1 * 1
%%% Yt         : Label matrix of target domain, n2 * m2 * 1
%%%%% win      : Window size of spatial filter of SSWK-MEDA,
%%%%%            win = 1 means 3*3, win = 2 means 5*5, 
%%%%%            win = 3 means 7*7, win = 4 means 9*9
%% Outputs:
%%%% Xs_spa    : Spatial feature matrix of source domain, n1 * m1 * d
%%%% Xt_spa    : Spatial feature matrix of target domain, n2 * m2 * d

% Xs
  [axs,bxs,cxs]=size(Xs_spe);
  Xs_spa=Xs_spe;
    
  for i = 1:axs
      for j = 1:bxs
          weights = 0;  
          for  p = -win:win
              for  q = -win:win
                  if  i+p>0 && j+q>0 && i+p<=axs && j+q<=bxs && Ys(i,j)~=0 &&  Ys(i+p,j+q)~=0
                       Xs_spa(i,j,:) = Xs_spa(i,j,:) + Xs_spe(i+p,j+q,:) ;
                       weights = weights + 1;
                  end
              end
          end
          Xs_spa(i,j,:) = (Xs_spa(i,j,:) - 2 * Xs_spe(i,j,:)) ./ (weights - 1);
      end
  end   
  
% Xt
[axt,bxt,cxt]=size(Xt_spe);
Xt_spa=Xt_spe;
  
  for i = 1:axt
      for j = 1:bxt
          weights = 0;  
          for  p = -win:win
              for  q = -win:win
                  if  i+p>0 && j+q>0 && i+p<=axt && j+q<=bxt && Yt(i,j)~=0 &&  Yt(i+p,j+q)~=0
                       Xt_spa(i,j,:) = Xt_spa(i,j,:) + Xt_spe(i+p,j+q,:) ;
                       weights = weights + 1;
                  end
              end
          end
          Xt_spa(i,j,:) = (Xt_spa(i,j,:) - 2 * Xt_spe(i,j,:)) ./ (weights - 1);
      end
  end   
  
end