function ISIplot(choiceSeq, saveDir)

% Hongli Wang, 2/26/2019
% plot the distribution of interswitch intervals
% input: choiceSeq (number-choice pair not matter as long as contigent

close all;

plotDefault;

% count the ISI
switchInd = [];
for choice = 1:length(choiceSeq)-1
    if choiceSeq(choice+1) ~= choiceSeq(choice)
        switchInd = [switchInd, choice];
    end
end
ISI = diff(switchInd);

% fit exponential
Stats = tabulate(ISI);
value = Stats(:,1); prob = Stats(:,3)/100;

% single term
[curve_exp1, goodness_exp1] = fit(value, prob , 'exp1');

%double term
[curve_exp2, goodness_exp2] = fit(value, prob, 'exp2');

% triple term
[curve_exp3, goodness_exp3] = fit(value, prob, 'a*exp(b*x) + c*exp(d*x) + e*exp(f*x)', 'Startpoint', [1, -0.8, 0.1, -0.1, 0, -0.01]);

% get pdf for ISI

%plot the single/double term exponential

figure;
yyaxis left
h=histogram(ISI, max(ISI));
h.FaceColor = [0 0 0];
ylabel('Occurance');
yyaxis right
plot(curve_exp1,'black');hold on; plot(curve_exp2);
xlabel('ISI');
ylabel('Probability');
legend('ISI distribution','1 expo','2 expos');
%save figures
print(gcf,'-dpng',[saveDir,'/ISI_distribution']);    %png format
saveas(gcf, [saveDir,'/ISI_distribution'], 'fig');   

% calculate the logliklihood
log_Lexp1 = 0; log_Lexp2 = 0; log_Lexp3 = 0;
for iter = 1:length(ISI)
    if ISI(iter) == 0
        log_tempexp1 = realmin;
        log_tempexp2 = realmin;
        log_texpexp3 = realmin;
    else
        log_tempexp1 = log(curve_exp1(ISI(iter)));
        log_tempexp2 = log(curve_exp2(ISI(iter)));
        log_texpexp3 = log(curve_exp3(ISI(iter)));
    end
    log_Lexp1 = log_Lexp1 + log_tempexp1;
    log_Lexp2 = log_Lexp2 + log_tempexp2;
    log_Lexp3 = log_Lexp3 + log_texpexp3;
end

figure; plot([1,2,3], [log_Lexp1, log_Lexp2, log_Lexp3], 'black', 'Linewidth', 2 );
xlabel('Number of components');
ylabel('Log likelihood');
xticks([1 2 3]);
%save the figures
print(gcf,'-dpng',[saveDir,'/ISI_expo_loglikelihood']);    %png format
saveas(gcf, [saveDir,'/ISI_expo_loglikelihood'], 'fig');   
