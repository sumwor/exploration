function [qpar, negloglike, bic, nlike, hess]=fit_Explore_priorsV(stats,fit_fun,initpar,x,lb,ub)    
% % fit_fun %
%PURPOSE:   Fit the choice behavior to a model using maximum likelihood estimate
%AUTHORS:   AC Kwan 170518
%
%INPUT ARGUMENTS
%   stats:      stats of the game
%   fit_fun:    the model to fit, e.g., Q_RPEfun
%   initpar:    initial values for the parameters
%   x:          which player? (1 or 2)
%   lb:         lower bound values for the parameters (if used)
%   ub:         upper bound values for the parameters (if used)
%
%OUTPUT ARGUMENTS
%   qpar:       extracted parameters
%   negloglike: negative log likelihood
%   bic:        Bayesian information criterion
%   nlike:      normalized likelihood

%%
%%
maxit=1e6;
maxeval=1e6;
op=optimset('fminsearch');
op.MaxIter=maxit;
op.MaxFunEvals=maxeval;

c = stats.c;
% r = stats.r(:,x); no need for reward
r = stats.r;
bin = stats.bin;
% V = stats.ql - stats.qr;
% RU = stats.sigmaL - stats.sigmaR;
% V_TU = V ./ (sqrt(stats.sigmaL.^2 + stats.sigmaR.^2));
% if strcmp(stats.playerlabel{1},'algo_UCB')
%     model = ones(size(c)); 
% elseif strcmp(stats.playerlabel{1}, 'algo_Thompson')
%     model = ones(size(c)) * 2;
% else
%     model = ones(size(c)) * 3;
% end

func_handle = str2func(fit_fun);
if ~exist('lb','var')
    % use fminunc to get a sd estimation
    [qpar,negloglike,exitflag,out,grad,hess]=fminunc(func_handle, initpar, op, [c r],bin);
else
    [qpar,negloglike,exitflag,out,lambda,grad,hess]=fmincon(func_handle, initpar, [], [], [], [], lb, ub, [], op, [c r],bin);
end

if exitflag==0
    qpar=nan(size(qpar));   %did not converge to a solution, so return NaN
    negloglike=nan;
end

%% BIC, bayesian information criterion
%BIC = -logL + klogN
%L = negative log-likelihood, k = number of parameters, N = number of trials
%larger BIC value is worse because more parameters is worse, obviously
bic = negloglike + numel(initpar)*log(sum(~isnan(c)));

%% Normalized likelihood 
%(Ito and Doya, PLoS Comp Biol, 2015)
nloglike = -negloglike/sum(~isnan(c));
nlike = exp(nloglike);

end