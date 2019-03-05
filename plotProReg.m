function plotProReg(weight, hess, tag, saveDir)

% Hongli Wang 02/26/2019
% plot the coefficient of probit regression
% input:        weight: regression coefficient w1,w2,w3
%               hess: hessian matrix of MLE
%               tag: struct with regression information
%               tag.var: 1:{'V','RU','V/TU'} or 2:{'V','RU','1/TU'}
%               tag.title: keywords in title
%               saveDir: path to save the figures
plotDefault;

fullstErr = sqrt(diag(inv(hess)));
stErr=fullstErr(1:3);
% plot the result
figure;
bar(weight, 'black');
hold on;errorbar([1,2,3],weight,stErr,'.','LineWidth',2,'color','black');
ylabel('Coefficient');
if tag.var == 1
    xticklabels({'V','RU','V/TU'});
else
    xticklabels({'V','RU','1/TU'});
end
titleTxt = ['Regression coefficient for animal data ',tag.title];
title(titleTxt);

%save the figures
print(gcf,'-dpng',[saveDir,'/',titleTxt]);    %png format
saveas(gcf, [saveDir,'/',titleTxt], 'fig');   