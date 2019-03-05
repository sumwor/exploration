% hybrid regression model for UCB and Thompson sampling

function like0=BTHfun_singleV(xpar,dat)
%PURPOSE:   Function for maximum likelihood estimation, called by
%           fit_qlearn().
%
%INPUT ARGUMENTS
%   xpar:       w1, w2, w3, model
%               w1,w2,w3: coefficient for different factors; model: hybrid
%               or UCB or Thompson
%   dat:        data
%               dat(:,1) = choice vector
%               dat(:,2) = V
%               dat(:,3) = RU
%               dat(:,4) = V/TU

%OUTPUT ARGUMENTS
%   like0:      the log-likelihood

w1 = xpar(1);
w2 = xpar(2);
w3 = xpar(3);

nt=size(dat,1);
like0=0;
V = dat(:,2);
RU = dat(:,3);
inv_TU = dat(:,4);
for k = 1:nt
    pleft = normcdf(w1*V(k) + w2 * RU(k) + w3*inv_TU(k));
    pright=1-pleft;
        
    if pright==0, pright=realmin; end;        % Smallest positive normalized floating point number
    if pleft==0, pleft=realmin; end;            
  
    if dat(k,1)>0, logp=log(pright);
    else logp=log(pleft);
    end  
 
    like0=like0-logp;  % calculate log likelihood
end
end
