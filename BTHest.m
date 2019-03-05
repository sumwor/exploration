function [ql, qr, sigmaL, sigmaR]=BTHest(par,choice,reward,bin)

% input:  choice: animal choice history
%         reward: animal reward history
%         par: prior parameters
% output
%       V: estimated action value based on bayes with beta prior
%       RU: estimated sigmreward-sigmaR
%       TU: estimated sqrt(sigmreward^2+sigmaR^2);

prior.aL = par(1);
prior.bL = par(2);
prior.aR = par(3);
prior.bR = par(4);
% bin = int(par(5));
ql = zeros(length(choice), 1);qr = zeros(length(choice), 1);
sigmaL = zeros(length(choice), 1);sigmaR = zeros(length(choice),1);

%aL = 1; bL = 1; aR = 1; bR = 1; % beta prior parameters

for iter = 1:length(choice)
    if iter == 1  %if this is the first trial
        ql(iter) = prior.aL/(prior.aL+prior.bL);
        qr(iter) = prior.aR/(prior.aR+prior.bR);
    % rewardpe(iter,x) = NaN;
        % pl(iter) = 0.5;
        sigmaL(iter) = sqrt( prior.aL* prior.bL / (( prior.aL+ prior.bL)^2*( prior.aL+ prior.bL+1)));
        sigmaR(iter) = sqrt( prior.aR* prior.bR / (( prior.aR+ prior.bR)^2*( prior.aR+ prior.bR+1)));
    %aL(iter,x) = reward;
    %bL(iter,x) = bL;
    %aR(iter,x) = aR;
    %bR(iter,x) = bR;
    %% gershman paper
    % sigmreward(iter,x) = sqrt(params.sigma2(1));
    % sigmaR(iter,x) = sqrt(params.sigma2(2));
    % stats.alphreward(iter,x) = sigmreward(iter,x)^2/(sigmreward(iter,x)^2+params.tau2(1));
    % stats.alphaR(iter,x) = sigmaR(iter,x)^2/(sigmaR(iter,x)^2 + params.tau2(2));
    elseif iter <= bin
        LeftRew = sum(choice(1:iter)==-1 & reward(1:iter)==1);
        LeftNRew = sum(choice(1:iter)==-1 & reward(1:iter)==0);
        RightRew = sum(choice(1:iter)==1 & reward(1:iter)==1);
        RightNRew = sum(choice(1:iter)==1 & reward(1:iter)==0);
        aL = prior.aL+LeftRew;
        bL = prior.bL+LeftNRew;
        aR = prior.aR+RightRew;
        bR = prior.bR+RightNRew;
    %update posterior
        ql(iter) = (aL+LeftRew)/(aL+bL+LeftRew+LeftNRew);
        qr(iter) = (aR+RightRew)/(aR+bR+RightRew+RightRew);
        sigmaL(iter) = sqrt((aL+LeftRew)*(bL+LeftNRew) / ((aL+LeftRew+bL+LeftNRew)^2*(aL+bL+1+LeftRew+LeftNRew)));
        sigmaR(iter) = sqrt((aR+RightRew)*(bR+RightNRew) / ((aR+RightRew+bR+RightNRew)^2*(aR+RightRew+bR+RightNRew+1)));
    else % only take in most recent 10 trials
        LeftRew = sum(choice(iter-bin:iter-1)==-1 & reward(iter-bin:iter-1)==1);
        LeftNRew = sum(choice(iter-bin:iter-1)==-1 & reward(iter-bin:iter-1)==0);
        RightRew = sum(choice(iter-bin:iter-1)==1 & reward(iter-bin:iter-1)==1);
        RightNRew = sum(choice(iter-bin:iter-1)==1 & reward(iter-bin:iter-1)==0);
        aL = prior.aL+LeftRew;
        bL = prior.bL+LeftNRew;
        aR = prior.aR+RightRew;
        bR = prior.bR+RightNRew;
        %update posterior
        ql(iter) = aL/(aL+bL);
        qr(iter) = aR/(aR+bR);
        sigmaL(iter) = sqrt(aL*bL / ((aL+bL)^2*(aL+bL)));
        sigmaR(iter) = sqrt(aR*bR / ((aR+bR)^2*(aR+bR)));
    end
end