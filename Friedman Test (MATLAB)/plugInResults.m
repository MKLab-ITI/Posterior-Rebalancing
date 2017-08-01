close all;

imported = import_output('results.txt');

GM = imported(:,[1 5 9 13])';
Imb = imported(:,[2 6 10 14])';
AUC = imported(:,[3 7 11 15])';
ILoss = imported(:,[4 8 12 16])';

labels = {'TIR', 'TPlugIn', 'RMT'};
p = [0.05];

figure(1);
friedman_test(-GM(2:4,:)',labels,p,'GM');
figure(2);
friedman_test(Imb(2:4,:)',labels,p,'Imb');
figure(3);
friedman_test(-AUC(2:4,:)',labels,p,'AUC');
figure(4);
friedman_test(ILoss(2:4,:)',labels,p,'ILoss');

imported = import_output('resultsSampling.txt');

GM = imported(:,[1 5 9 13])';
Imb = imported(:,[2 6 10 14])';
AUC = imported(:,[3 7 11 15])';
ILoss = imported(:,[4 8 12 16])';

labels = {'Sampling', 'Sampling+TIR', 'Sampling+TPlugIn', 'Sampling+RMT'};
p = [0.1];

figure(5);
friedman_test(-GM(1:4,:)',labels,p,'GM');
figure(6);
friedman_test(Imb(1:4,:)',labels,p,'Imb');
figure(7);
friedman_test(-AUC(1:4,:)',labels,p,'AUC');
figure(8);
friedman_test(ILoss(1:4,:)',labels,p,'ILoss');