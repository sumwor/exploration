clear all;
close all;
plotDefault;
%% load data file

base_dir = 'E:\data\bandit_logfile\752\bandit_new\data';
save_dir = 'E:\data\exploration';

cd 'E:\data\bandit_logfile\752\bandit_new\data'
D = dir();
animalsChoice = [];
animalsOutcome = [];
animalsResponse = [];
block = [];
for iter = 1:length(D)
    if ~isempty(strfind(D(iter).name,'.log')) 
        logfile = D(iter).name;
        [ logData ] = parseLogfileHW(base_dir, logfile);
        [ sessionData, trialData] = banditHW_getSessionData( logData );
        animalsChoice = [animalsChoice; trialData.response(trialData.response~=0)];
        animalsOutcome = [animalsOutcome; trialData.outcome(trialData.response~=0)];
        animalsResponse = [animalsResponse; trialData.rt(trialData.response~=0)];
        block = [block; trialData.start(trialData.response~=0)];
    end
end

%proces the outcome
OUTCOME.LCR = 4;        % left correct reward
OUTCOME.LIR = 40;       % left incorrect reward
OUTCOME.RCR = 8;        % right correct reward 
OUTCOME.RIR = 80;       % right incorrect reward 
OUTCOME.LIN = 100;       % left incorrect no reward
OUTCOME.LCN = 101;       % left correct no reward
OUTCOME.RCN = 111;       % right correct no reward
OUTCOME.RIN = 110;       % right incorrect no reward

% get reward
animalsReward = zeros(size(animalsOutcome));
animalsReward(animalsOutcome == 4 | animalsOutcome == 8 | animalsOutcome == 40 | animalsOutcome == 80) = 1;

%% plot the ISI
ISIplot(animalsChoice,save_dir);

%% fit the hmm model
choiceSeq = double((animalsChoice-1)');
[estimateStates, hmmF] = HMMest(choiceSeq);

%% try UCB and Thompson exploration

stats.c = ((choiceSeq-1.5)*2)';
stats.r = animalsReward;

%% set prior distribution to alphaL=betaL=alphaR=betaR = 1
priorPar = [1,1,1,1];
[stats.ql,stats.qr,stats.sigmaL,stats.sigmaR] = BTHest(priorPar, stats.c,stats.r,5);

% fit the hybrid model
% fit params: w1, w2, w3
% fixed params: alphaL=1; betaL=1;alphaR = 1; betaR = 1;
fun = 'BTHfun';
initpar = [1, 1, 1];
[fitpar.BTH, ~, bic.BTH, nlike.BTH, hess.BTH]=fit_Explore(stats,fun,initpar,1);

tag.var =1;
tag.title ='fixed prior';
plotProReg(fitpar.BTH, hess.BTH, tag, save_dir);

% estimate posterior uncertainty
% correlation between value estimation of hybrid model and value estimation
% of RL-RPE model is 0.4360

%% fit the prior parameters as well

% can we include the prior parameters as variable and get a MLE estimation?
% yes we can
% fit the hybrid model
fun = 'BTHfun_prior';
initpar = [1, 1, 1, 1, 1, 1, 1];
lb = [-inf, -inf, -inf, 0,0,0,0];
ub = [];
stats.bin = 5;

% try different bin size, results skeptical
fitpar.BTH_prior_Bin = cell(1,10);
bic.BTH_prior_Bin = zeros(1,10);
nlike.BTH_prior_Bin = zeros(1,10);
hess.prior_Bin = cell(1,10);
for bin = 1:10
    stats.bin = bin;
    [t_fitpar, ~, t_bic, t_nlike, t_hess]=fit_Explore_prior(stats,fun,initpar,1,lb,ub);
    fitpar.BTH_prior_Bin{bin} = t_fitpar;
    bic.BTH_prior_Bin(bin) = t_bic;
    nlike.BTH_prior_Bin(bin) = t_nlike;
    hess.prior_Bin{bin} = t_hess;
end
% the best model is bin=2, probability related to the exploit trials
% fit the model on exploration trials only? 

% plot the result for bin=5
tag.var =1;
hess_5 = hess.prior_Bin{5};
weight_5 = fitpar.BTH_prior_Bin{5}(1:3);
tag.title ='fit prior';
plotProReg(weight_5, hess_5, tag, save_dir);

%% separate trials into exploration and exploitation (HMM model results)

% do the regression use exploration trials (masked by HMM model only)
%ql,qr, sigmaL,sigmaR estimated based on fixed prior (1,1,1,1)
exploreStats.c = stats.c(estimateStates==1);
exploreStats.r = stats.r(estimateStates==1);
exploreStats.ql = stats.ql(estimateStates==1);
exploreStats.qr = stats.qr(estimateStates==1);
exploreStats.sigmaL = stats.sigmaL(estimateStates==1);
exploreStats.sigmaR = stats.sigmaR(estimateStates==1);

% regression: fixed prior, exploration
fun = 'BTHfun';
initpar = [1, 1, 1];
[fitpar.BTH_ex, ~, bic.BTH_ex, nlike.BTH_ex, hess.ex]=fit_Explore(exploreStats,fun,initpar,1);

%plot the figure
tag.var =1;
tag.title ='fixed prior & explore';
plotProReg(fitpar.BTH_ex, hess.ex, tag, save_dir);


% apply UCB+Thompson to explore trials only
% result is problematic (too few trials maybe)
% check for bug 2/27/2019
% there's a problem for the fitting, although current trials should all be
% explore, however, previous trials to update the belief need to take
% exploitation into account

%regression: fit prior, exploration
fun = 'BTHfun_prior';
initpar = [1, 1, 1, 1, 1, 1, 1];
lb = [-inf, -inf, -inf, 0,0,0,0];
ub = [];
exploreStats.bin = 5;
[fitpar.BTH_prior_ex, ~,bic.BTH_prior_ex, nlike.BTH_prior_ex , hess.prior_ex]=fit_Explore_prior(exploreStats,fun,initpar,1,lb,ub);
%plot the figure
tag.var =1;
weight_prior_ex = fitpar.BTH_prior_ex(1:3);
tag.title ='fit prior & explore';
plotProReg(weight_prior_ex, hess.prior_ex, tag, save_dir);


%fit for another regression with single V in it
fun = 'BTHfun_singleV';
initpar = [1, 1, 1];
[fitpar.BTH_sV, ~, bic.BTH_sV, nlike.BTH_sV, hess.sV]=fit_Explore_singleV(stats,fun,initpar,1);
tag.var =2;
tag.title ='fixed prior & invTU';
plotProReg(fitpar.BTH_sV, hess.sV, tag, save_dir);


%fit for exploration trials with single V
fun = 'BTHfun_singleV';
initpar = [1, 1, 1];
[fitpar.BTH_exsV, ~, bic.BTH_exsV, nlike.BTH_exsV, hess.exsV]=fit_Explore_singleV(exploreStats,fun,initpar,1);
tag.var =2;
tag.title ='fixed prior & invTU & explore';
plotProReg(fitpar.BTH_exsV, hess.exsV, tag, save_dir);

%fit for exploration trials with V/TU
fun = 'BTHfun_singleV';
initpar = [1, 1, 1];
[fitpar.BTH_ex, ~, bic.BTH_ex, nlike.BTH_ex, hess.ex]=fit_Explore(exploreStats,fun,initpar,1);
tag.var =1;
tag.title ='fixed prior & V/TU & explore';
plotProReg(fitpar.BTH_ex, hess.ex, tag, save_dir);

%fit the exploit
exploitStats.c = stats.c(estimateStates~=1);
exploitStats.r = stats.r(estimateStates~=1);
exploitStats.ql = stats.ql(estimateStates~=1);
exploitStats.qr = stats.qr(estimateStates~=1);
exploitStats.sigmaL = stats.sigmaL(estimateStates~=1);
exploitStats.sigmaR = stats.sigmaR(estimateStates~=1);
exploitStats.bin = 5;

initpar = [1, 1, 1];
[fitpar.BTH_extsV, ~, bic.BTH_extsV, nlike.BTH_extsV, hess.extsV]=fit_Explore_singleV(exploitStats,fun,initpar,1);
tag.var =2;
tag.title ='fixed prior & invTU & exploit';
plotProReg(fitpar.BTH_extsV, hess.extsV, tag, save_dir);

% fit prior for exploration trials
% exploration trials are too small to fit?
fun = 'BTHfun_sV';
initpar = [1, 1, 1, 1, 1, 1, 1];
lb = [-inf, -inf, -inf, 0,0,0,0];
ub = [];
exploreStats.bin = 5;
[fitpar.BTH_prior_exsV, ~,bic.BTH_prior_exsV, nlike.BTH_prior_exsV , hess.prior_exsV]=fit_Explore_prior(exploreStats,fun,initpar,1,lb,ub);
%plot the figure
tag.var =2;
weight_prior_exsV = fitpar.BTH_prior_exsV(1:3);
tag.title ='fit prior & explore';
plotProReg(weight_prior_exsV, hess.prior_exsV, tag, save_dir);
%% using previously estimated prior (but these are estimated for the full dataset
% potentially problemetic

par_est = fitpar.BTH_prior_Bin{5}(4:7);
bin = 5;
[ql_est, qr_est, sigmaL_est, sigmaR_est]=BTHest(par_est,stats.c,stats.r,bin);

% exploit regression with prior estimated from data
exploitStatsPrior.c = stats.c(estimateStates~=1);
exploitStatsPrior.r = stats.r(estimateStates~=1);
exploitStatsPrior.ql = ql_est(estimateStates~=1);
exploitStatsPrior.qr = qr_est(estimateStates~=1);
exploitStatsPrior.sigmaL = sigmaL_est(estimateStates~=1);
exploitStatsPrior.sigmaR = sigmaR_est(estimateStates~=1);
exploitStatsPrior.bin = 5;

fun = 'BTHfun_singleV';
initpar = [1, 1, 1];
[fitpar.BTH_extsVprior, ~, bic.BTH_extsVprior, nlike.BTH_extsVprior, hess.extsVprior]=fit_Explore_singleV(exploitStatsPrior,fun,initpar,1);
%plot the figure
tag.var =2;
tag.title ='fit prior(full data) & exploit & invTU';
plotProReg(fitpar.BTH_extsVprior, hess.extsVprior, tag, save_dir);

%exploration regressions with prior estimated from data
exploreStatsPrior.c = stats.c(estimateStates==1);
exploreStatsPrior.r = stats.r(estimateStates==1);
exploreStatsPrior.ql = stats.ql(estimateStates==1);
exploreStatsPrior.qr = stats.qr(estimateStates==1);
exploreStatsPrior.sigmaL = stats.sigmaL(estimateStates==1);
exploreStatsPrior.sigmaR = stats.sigmaR(estimateStates==1);
%fit for another regression with single V in it
fun = 'BTHfun_singleV';
initpar = [1, 1, 1];
[fitpar.BTH_exsVprior, ~, bic.BTH_exsVprior, nlike.BTH_exsVprior, hess.exsVprior]=fit_Explore_singleV(exploreStatsPrior,fun,initpar,1);
%plot the figure
tag.var =2;
tag.title ='fit prior(full data) & explore & invTU';
plotProReg(fitpar.BTH_exsVprior, hess.exsVprior, tag, save_dir);

%% estimate priors for exploration trials only (but using all trials as likelihood
% in exploitation: softmax decision rule
%error bar problematic
fun = 'BTHfun_HMM';
stats.state = estimateStates';
stats.bin=5;
initpar = [1, 1, 1,1,1,1,1];
lb = [-inf, -inf, -inf, 0,0,0,0];
ub = [];
[fitpar.BTH_exHMM, ~, bic.BTH_exHMM, nlike.BTH_exHMM, hess.exHMM]=fit_Explore_HMMprior(stats,fun,initpar,1, lb,ub);
%plot the figure
tag.var =2;
weight_HMM = fitpar.BTH_exHMM(1:3);
tag.title ='fit prior & explore & HMM';
plotProReg(weight_HMM, hess.exHMM, tag, save_dir);
%the explore trials has a huge bias -> the mean of beta prior for right is close to 1.  
%calculate log like for exploitation (greedy), no change
%calculate loglike for exploitation using softmax, results okay yet the
%hess matrix is problematic
%% response time with uncertainty
% estimate RU, TU using fit par (this part using fit prior distribution
% from whole data
% instead of [1,1,1,1]

TU = sqrt(sigmaL_est.^2+sigmaR_est.^2);
RU = sigmaL_est-sigmaR_est;
LeftRt = animalsResponse(stats.c==-1); RightRt = animalsResponse(stats.c==1);
TUleft = TU(stats.c==-1);TUright = TU(stats.c==1);
RUleft = RU(stats.c==-1); RUright = RU(stats.c==1);
figure;
subplot(3,2,1)
scatter(TU,animalsResponse);
xlabel('TU'); ylabel('Rt (s)');
title('TU vs total response time')
subplot(3,2,2)
scatter(RU, animalsResponse);
xlabel('RU');ylabel('Rt (s)');
title('RU vs total response time');
subplot(3,2,3)
scatter(TUleft, LeftRt);
xlabel('TU');ylabel('Rt (s)');
title('TU vs left response time');
subplot(3,2,4)
scatter(TUright,RightRt);
xlabel('TU');ylabel('Rt (s)');
title('TU vs right response time');
subplot(3,2,5)
scatter(RUleft,LeftRt);
xlabel('RU');ylabel('Rt (s)');
title('RU vs left response time');
subplot(3,2,6)
scatter(RUright,RightRt);
xlabel('RU');ylabel('Rt (s)');
title('RU vs left response time');
%save the figures

titleTxt = 'rt left or right vs RU and TU'; 
print(gcf,'-dpng',[save_dir,'/',titleTxt]);    %png format
saveas(gcf, [save_dir,'/',titleTxt], 'fig');   

%now try explore vs exploit and response time
TU_exploreLeft = TU(estimateStates==1 & stats.c'==-1);
TU_exploreRight = TU(estimateStates==1 & stats.c'==1);
TU_exploitLeft = TU(estimateStates==2);
TU_exploitRight = TU(estimateStates==3);
rt_exploreLeft = animalsResponse(estimateStates==1 & stats.c'==-1);
rt_exploreRight = animalsResponse(estimateStates==1 & stats.c'==1);
rt_exploitLeft = animalsResponse(estimateStates==2);
rt_exploitRight = animalsResponse(estimateStates==3);
figure;
subplot(3,2,1)
scatter(TU_exploreLeft,rt_exploreLeft);
hold on; scatter(TU_exploreRight,rt_exploreRight);
title('Explore left and right');
subplot(3,2,2);
scatter(TU_exploitLeft,rt_exploitLeft);
hold on; scatter(TU_exploitRight, rt_exploitRight);
title('Exploit left and right');
subplot(3,2,3)
scatter(TU_exploreLeft,rt_exploreLeft);
title('Explore left');
subplot(3,2,4);
scatter(TU_exploreRight, rt_exploreRight);
title('Explore right')
subplot(3,2,5)
scatter(TU_exploitLeft, rt_exploitLeft);
title('Exploit left');
xlabel('TU');
subplot(3,2,6)
scatter(TU_exploitRight, rt_exploitRight);
title('Exploit right');
xlabel('TU');
%save figures
titleTxt = 'rt explore or exploit or left or right vs TU'; 
print(gcf,'-dpng',[save_dir,'/',titleTxt]);    %png format
saveas(gcf, [save_dir,'/',titleTxt], 'fig');   

%%
fun = 'Q_RPEfun';
initpar=[0.5 10]; % initial [alpha beta]
[fitpar.Q_RPE, ~, bic.Q_RPE, nlike.Q_RPE]=fit_fun(stats,fun,initpar,1)
% newstats.c= stats.c; newstats.r = stats.r;
% [fitpar.Q_RPE1, ~, bic.Q_RPE1, nlike.Q_RPE1]=fit_fun(newstats,fun,initpar,1);
fun = 'DFQfun';
initpar=[0.1 0 0.8 0]; % initial [alpha1 alpha2 kappa1 kappa2]
[fitpar.DFQ, ~, bic.DFQ, nlike.DFQ]=fit_fun(stats,fun,initpar,1);

fun = 'FQfun';
initpar=[0.1 0.8 0]; % initial [alpha1 kappa1 kappa2]
[fitpar.FQ, ~, bic.FQ, nlike.FQ]=fit_fun(stats,fun,initpar,1);

fun = 'Qfun';
initpar=[0.1 0.8 0]; % initial [alpha1 kappa1 kappa2]
[fitpar.Q, ~, bic.Q, nlike.Q]=fit_fun(stats,fun,initpar,1);

% [~, ~, bic.logregCRInt2, nlike.logregCRInt2]=logreg_CRInt(stats,1,2);
% [~, ~, bic.logregCRInt5, nlike.logregCRInt5]=logreg_CRInt(stats,1,5);
% [~, ~, bic.logregCRInt10, nlike.logregCRInt10]=logreg_CRInt(stats,1,10);

fitpar
bic

% plot_session_qparam(stats,1, 2000);

%% estimate action value for different RL algorithm
% use the fit par to estimate action value
AValueRPE = zeros(2, length(choiceSeq));
AValueQ = zeros(2, length(choiceSeq));
AValueFQ = zeros(2, length(choiceSeq));
AValueDFQ = zeros(2, length(choiceSeq));
VRtempRPE = 0; VLtempRPE = 0;
VRtempQ = 0; VLtempQ = 0;
VRtempFQ = 0; VLtempFQ = 0;
VRtempDFQ = 0; VLtempDFQ = 0;
for iter = 1: length(choiceSeq)
    %update action value
    [VRtempRPE, VLtempRPE] = updateActionValueRPE(VRtempRPE, VLtempRPE, stats.c(iter),stats.r(iter), fitpar.Q_RPE(1));
    [VRtempQ, VLtempQ] = updateActionValue(VRtempQ, VLtempQ, stats.c(iter), stats.r(iter), fitpar.Q(1),0,fitpar.Q(2), fitpar.Q(3));
    [VRtempFQ, VLtempFQ] = updateActionValue(VRtempFQ, VLtempFQ, stats.c(iter), stats.r(iter), fitpar.FQ(1), fitpar.FQ(1),fitpar.FQ(2), fitpar.FQ(3));
    [VRtempDFQ, VLtempDFQ] = updateActionValue(VRtempDFQ, VLtempDFQ, stats.c(iter), stats.r(iter), fitpar.DFQ(1), fitpar.DFQ(2), fitpar.DFQ(3), fitpar.DFQ(4));
    AValueRPE(:,iter) = [VRtempRPE, VLtempRPE];
    AValueQ(:,iter) = [VRtempQ, VLtempQ];
    AValueFQ(:,iter) = [VRtempFQ, VLtempFQ];
    AValueDFQ(:,iter) = [VRtempDFQ, VLtempDFQ];
end

% plot the choice prob 
figure;
subplot(2,2,1)
plotValue((AValueRPE(1,:)-AValueRPE(2,:)), stats.c, 'Q-RPE');
subplot(2,2,2)
plotValue((AValueQ(1,:)-AValueQ(2,:)), stats.c, 'Q');
subplot(2,2,3)
plotValue((AValueFQ(1,:)-AValueFQ(2,:)), stats.c, 'FQ');
subplot(2,2,4)
plotValue((AValueDFQ(1,:)-AValueDFQ(2,:)), stats.c, 'DFQ');

% for hmm estimate
exploreHmm = (estimateStates ==1);



%% mark the trials by explore/exploit based on action values
% trials that choose a low value side is exploration
exploreRPE = double((AValueRPE(1)>AValueRPE(2)&stats.c==-1) | (AValueRPE(1)<AValueRPE(2)&stats.c==1));
exploreQ = (AValueQ(1)>AValueQ(2)&stats.c==-1) | (AValueQ(1)<AValueQ(2)&stats.c==1);
exploreFQ = (AValueFQ(1)>AValueFQ(2)&stats.c==-1) | (AValueFQ(1)<AValueFQ(2)&stats.c==1);
exploreDFQ = (AValueDFQ(1)>AValueDFQ(2)&stats.c==-1) | (AValueDFQ(1)<AValueDFQ(2)&stats.c==1);
figure;
plot((exploreRPE(1:100)), 'LineWidth',2);
hold on;plot(exploreQ(1:100), 'LineWidth',2);
hold on;plot(exploreFQ(1:100), 'LineWidth',2);
hold on;plot(exploreDFQ(1:100), 'LineWidth',2);
hold on; plot(exploreHmm(1:100), 'LineWidth',2);


%% plot response time versus the exploration

% HMM estimate
rtExplore_Hmm = animalsResponse(estimateStates == 1);
rtExploit_HmmLeft = animalsResponse(estimateStates == 2);
rtExploit_HmmRight = animalsResponse(estimateStates == 3);
err = [sqrt(var(rtExplore_Hmm)), sqrt(var([rtExploit_HmmLeft;rtExploit_HmmRight]))];
figure; bar([mean(rtExplore_Hmm), mean([rtExploit_HmmLeft;rtExploit_HmmRight])]);
hold on; errorbar([mean(rtExplore_Hmm), mean([rtExploit_HmmLeft;rtExploit_HmmRight])],err,'bx');
% RL
rtExplore_QRPE = animalsResponse(logical(exploreRPE));
rtExploit_QRPE = animalsResponse(~logical(exploreRPE));

% exploration are the same for all RL algorithms

% figure;histogram(rtExplore_Hmm);
% hold on; histogram([rtExploit_Left; rtExploit_Right]);
function [Vr_t, Vl_t] = updateActionValue(Vr_s, Vl_s, A, R, a1,a2,k1,k2)
% update action value for DFQ, Q, FQ learning
    if A==1     % chose right
        if R==1  %reward
            Vr_t = (1-a1)*Vr_s+a1*k1;
            Vl_t = (1-a2)*Vl_s;
        elseif R == 0            %no reward
            Vr_t = (1-a1)*Vr_s-a1*k2;
            Vl_t = (1-a2)*Vl_s;
        end
    elseif A==-1    % choose left
        if R==1     %reward
            Vl_t = (1-a1)*Vl_s+a1*k1;
            Vr_t = (1-a2)*Vr_s;
        elseif R==0            %no reward
            Vl_t = (1-a1)*Vl_s-a1*k2;
            Vr_t = (1-a2)*Vr_s;
        end
    end   
end

function [Vr_t, Vl_t] = updateActionValueRPE(Vr_s, Vl_s, A, R, alpha)
    if A==1      %chose right
        Vr_t=Vr_s+alpha*(R-Vr_s);   
        Vl_t = Vl_s;
    elseif A==-1 %chose left
        Vl_t=Vr_s+alpha*(R-Vl_s);
        Vr_t = Vr_s;
    end   
end

function plotValue(v_difference, choice, algorithm)
% plot the value difference)
bin_size=0.01;
num=(round(max(v_difference)*100)/100+bin_size/2-(round(min(v_difference)*100)/100-bin_size/2))*100;
value_bin=zeros(1,num);
value_x=round(min(v_difference)*100)/100-bin_size/2:bin_size:round(max(v_difference)*100)/100-bin_size/2;

% xaxis = ceil(min(v_difference)*100)/100:bin_size:floor(max(v_difference)*100)/100;
pred_prob=1./(1+exp(-value_x));
for i=1:length(choice)
    ind=ceil((v_difference(i)-value_x(1))/bin_size);
    value_bin(ind)=value_bin(ind)+1;
end

%determine the real probability to choose right;
num_right=zeros(1,num);
for i=1:length(choice)
    index=ceil((v_difference(i)-value_x(1))/bin_size);
    if choice(i)==1 %if choose right
        num_right(index)=num_right(index)+1;
    end
end
prob_right=num_right./value_bin;
per_value=value_bin/sum(value_bin);

% figure;
yyaxis left; bar(value_x, per_value);
ylabel('Frequency');
hold on; yyaxis right; plot(value_x, pred_prob);
hold on; yyaxis right; plot(value_x, prob_right,'.', 'MarkerSize',25);
ylabel('P(right)');
xlabel('Action value difference (R-L)');

title(algorithm);
end

%% fit the UCB sampling parameters(lambda and gamma) to exploration data
    %no, you can actually infer theses from w1, w2
    %w1 = 1/lambda; w2 = gamma/lambda;
    
%% human data?