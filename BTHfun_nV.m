% hybrid regression model for UCB and Thompson sampling (as well as prior)

function like0=BTHfun_nV(xpar,dat,bin)
%PURPOSE:   Function for maximum likelihood estimation, called by
%           fit_qlearn().
%
%INPUT ARGUMENTS
%   xpar:       w1, w2,aL, bL, aR, bR
%               w1,w2: coefficient for different factors;
%               aL,bL,aR,bR: prior distribution parameters
%   dat:        data
%               dat(:,1) = choice vector
%               dat(:,2) = reward vector


%OUTPUT ARGUMENTS
%   like0:      the log-likelihood

w1 = xpar(1);
w2 = xpar(2);
prior.aL = xpar(3);
prior.bL = xpar(4);
prior.aR = xpar(5);
prior.bR = xpar(6);


nt=size(dat,1);
like0=0;
choice = dat(:,1);
reward = dat(:,2);
%V = dat(:,2);
%RU = dat(:,3);
%V_TU = dat(:,4);
par = [prior.aL,prior.bL, prior.aR, prior.bR];
[ql, qr, sigmaL, sigmaR] = BTHest(par,choice,reward,bin);
V = ql-qr;
RU = sigmaL - sigmaR;
V_TU = V ./ (sqrt(sigmaL.^2 + sigmaR.^2));
for k = 1:nt
    pleft = normcdf(w1 * RU(k) + w2*V_TU(k));
    pright=1-pleft;
        
    if pright==0, pright=realmin; end;        % Smallest positive normalized floating point number
    if pleft==0, pleft=realmin; end;            
  
    if dat(k,1)>0, logp=log(pright);
    else logp=log(pleft);
    end  
 
    like0=like0-logp;  % calculate log likelihood
end
end
