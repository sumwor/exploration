function stats=algo_Thompson(stats,params,x)

% % algo_Thompson %
%PURPOSE:   Simulate player based on Thompson sampling

%AUTHORS:   Hongli Wang 190218
%
%INPUT ARGUMENTS
%   stats:  stats of the game thus far
%   params: parameters that define the player's strategy
%       
%
%OUTPUT ARGUMENTS
%   stats:  updated with player's probability to choose left for next step

if stats.currTrial == 1  %if this is the first trial
    stats.ql(stats.currTrial,x) = params.aL/(params.aL+params.bL);
    stats.qr(stats.currTrial,x) = params.aR/(params.aR+params.bR);
    % stats.rpe(stats.currTrial,x) = NaN;
    stats.pl(stats.currTrial,x) = 0.5;
    stats.sigmaL(stats.currTrial,x) = sqrt(params.aL*params.bL / ((params.aL+params.bL)^2*(params.aL+params.bL+1)));
    stats.sigmaR(stats.currTrial,x) = sqrt(params.aR*params.bR / ((params.aR+params.bR)^2*(params.aR+params.bR+1)));
    %stats.aL(stats.currTrial,x) = params.aL;
    %stats.bL(stats.currTrial,x) = params.bL;
    %stats.aR(stats.currTrial,x) = params.aR;
    %stats.bR(stats.currTrial,x) = params.bR;
    %% gershman paper
    % stats.sigmaL(stats.currTrial,x) = sqrt(params.sigma2(1));
    % stats.sigmaR(stats.currTrial,x) = sqrt(params.sigma2(2));
    % stats.alphaL(stats.currTrial,x) = stats.sigmaL(stats.currTrial,x)^2/(stats.sigmaL(stats.currTrial,x)^2+params.tau2(1));
    % stats.alphaR(stats.currTrial,x) = stats.sigmaR(stats.currTrial,x)^2/(stats.sigmaR(stats.currTrial,x)^2 + params.tau2(2));
elseif stats.currTrial <= 6
    LeftRew = sum(stats.c==-1 & stats.r==1);
    LeftNRew = sum(stats.c==-1 & stats.r==0);
    RightRew = sum(stats.c==1 & stats.r==1);
    RightNRew = sum(stats.c==1 & stats.r==0);
    stats.aL = params.aL+LeftRew;
    stats.bL = params.bL+LeftNRew;
    stats.aR = params.aR+RightRew;
    stats.bR = params.bR+RightNRew;
    %update posterior
    stats.ql(stats.currTrial,x) = (params.aL+LeftRew)/(params.aL+params.bL+LeftRew+LeftNRew);
    stats.qr(stats.currTrial,x) = (params.aR+RightRew)/(params.aR+params.bR+RightRew+RightRew);
    stats.sigmaL(stats.currTrial,x) = sqrt((params.aL+LeftRew)*(params.bL+LeftNRew) / ((params.aL+LeftRew+params.bL+LeftNRew)^2*(params.aL+params.bL+1+LeftRew+LeftNRew)));
    stats.sigmaR(stats.currTrial,x) = sqrt((params.aR+RightRew)*(params.bR+RightNRew) / ((params.aR+RightRew+params.bR+RightNRew)^2*(params.aR+RightRew+params.bR+RightNRew+1)));
else % only take in most recent 10 trials
    LeftRew = sum(stats.c(stats.currTrial-5:stats.currTrial-1)==-1 & stats.r(stats.currTrial-5:stats.currTrial-1)==1);
    LeftNRew = sum(stats.c(stats.currTrial-5:stats.currTrial-1)==-1 & stats.r(stats.currTrial-5:stats.currTrial-1)==0);
    RightRew = sum(stats.c(stats.currTrial-5:stats.currTrial-1)==1 & stats.r(stats.currTrial-5:stats.currTrial-1)==1);
    RightNRew = sum(stats.c(stats.currTrial-5:stats.currTrial-1)==1 & stats.r(stats.currTrial-5:stats.currTrial-1)==0);
    stats.aL = params.aL+LeftRew;
    stats.bL = params.bL+LeftNRew;
    stats.aR = params.aR+RightRew;
    stats.bR = params.bR+RightNRew;
     %update posterior
    stats.ql(stats.currTrial,x) = stats.aL/(stats.aL+stats.bL);
    stats.qr(stats.currTrial,x) = stats.aR/(stats.aR+stats.bR);
    stats.sigmaL(stats.currTrial,x) = sqrt(stats.aL*stats.bL / ((stats.aL+stats.bL)^2*(stats.aL+stats.bL)));
    stats.sigmaR(stats.currTrial,x) = sqrt(stats.aR*stats.bR / ((stats.aR+stats.bR)^2*(stats.aR+stats.bR)));
end
    %% update action values and variance (Gershman paper)
    % stats.alphaL(stats.currTrial-1,x) = stats.sigmaL(stats.currTrial-1,x)^2/(stats.sigmaL(stats.currTrial-1,x)^2+params.tau2(1));
    % stats.alphaR(stats.currTrial-1,x) = stats.sigmaR(stats.currTrial-1,x)^2/(stats.sigmaR(stats.currTrial-1,x)^2 + params.tau2(2));
    % if stats.c(stats.currTrial-1,x)==-1     % if chose left on last trial
     %   stats.ql(stats.currTrial,x) = stats.ql(stats.currTrial-1,x) + stats.alphaL(stats.currTrial-1,x)*(stats.r(stats.currTrial-1,x)-stats.ql(stats.currTrial-1,x));
     %   stats.sigmaL(stats.currTrial,x) = sqrt(stats.sigmaL(stats.currTrial-1,x)^2*(1-stats.alphaL(stats.currTrial-1,x)));
     %   stats.qr(stats.currTrial,x)=stats.qr(stats.currTrial-1,x);    
     %   stats.sigmaR(stats.currTrial,x) = stats.sigmaR(stats.currTrial-1,x);
    % elseif stats.c(stats.currTrial-1,x)==1  % else, chose right
     %   stats.qr(stats.currTrial,x) = stats.qr(stats.currTrial-1,x) + stats.alphaR(stats.currTrial-1,x)*(stats.r(stats.currTrial-1,x)-stats.qr(stats.currTrial-1,x));
     %   stats.sigmaR(stats.currTrial,x) =  sqrt(stats.sigmaR(stats.currTrial-1,x)^2*(1-stats.alphaR(stats.currTrial-1,x)));
     %   stats.sigmaL(stats.currTrial,x) = stats.sigmaL(stats.currTrial-1,x);
     %   stats.ql(stats.currTrial,x)=stats.ql(stats.currTrial-1,x);
    % else            %no choice, then just hold on all latent variables
     %   stats.ql(stats.currTrial,x)=stats.ql(stats.currTrial-1,x);
     %   stats.qr(stats.currTrial,x)=stats.qr(stats.currTrial-1,x);
     %   stats.sigmaR(stats.currTrial,x) = stats.sigmaR(stats.currTrial-1,x);
     %   stats.sigmaL(stats.currTrial,x) = stats.sigmaL(stats.currTrial-1,x);
    %end
    
    %% UCB rule for action selection
    
stats.pl(stats.currTrial,x)=normcdf((stats.ql(stats.currTrial,x)-stats.qr(stats.currTrial,x))/sqrt(stats.sigmaL(stats.currTrial,x)^2+stats.sigmaR(stats.currTrial,x)^2));
    


end
