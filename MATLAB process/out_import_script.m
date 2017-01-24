filepaths = {'out/0.txt', ...
                'out/U1lin.txt', 'out/UTlin.txt', ...
                'out/U1inv.txt', 'out/UTinv.txt', ...
                'out/U1exp.txt', 'out/UTexp.txt', ...
                'out/1inv.txt', 'out/Tinv.txt', ...
                'out/1exp.txt', 'out/Texp.txt', ...
                'out/0+sample.txt', ...
                'out/U1lin+sample.txt', 'out/UTlin+sample.txt', ...
                'out/U1inv+sample.txt', 'out/UTinv+sample.txt', ...
                'out/U1exp+sample.txt', 'out/UTexp+sample.txt'};
labels = {'Base Classifier', ...
            '1lin', 'Tlin', ...
            '1inv', 'Tinv', ...
            '1exp', 'Texp', ...
            'L+1inv', 'L+Tinv', ...
            'L+1exp', 'L+Texp',...
            'Sampling', ...
            '1lin+Sampling', 'Tlin+Sampling', ...
            '1inv+Sampling', 'Tinv+Sampling', ...
            '1exp+Sampling', 'Texp+Sampling'};
        
%selected_labels = 1:7;suffix = '';
selected_labels = [1 2 4 12 13 15];suffix = '_sample';
p = 0.05;

%% import scores, fairness and auc for all classifiers
logistic_scores = [];
logistic_fairness = [];
logistic_auc = [];
sknn_scores = [];
sknn_fairness = [];
sknn_auc = [];
svm_scores = [];
svm_fairness = [];
svm_auc = [];
for i=1:length(filepaths)
    classifier = import_output(filepaths{i}, 2);
    logistic_scores = [logistic_scores;classifier(:,4)'];
    logistic_fairness = [logistic_fairness;classifier(:,3)'];
    logistic_auc = [logistic_auc;classifier(:,1)'];
    sknn_scores = [sknn_scores;classifier(:,8)'];
    sknn_fairness = [sknn_fairness;classifier(:,7)'];
    sknn_auc = [sknn_auc;classifier(:,5)'];
    svm_scores = [svm_scores;classifier(:,12)'];
    svm_fairness = [svm_fairness;classifier(:,11)'];
    svm_auc = [svm_auc;classifier(:,9)'];
end

%% find new labels
new_labels = {};
for i=1:length(selected_labels)
    new_labels{i} = labels{selected_labels(i)};
end
labels = new_labels;
logistic_scores = logistic_scores(selected_labels,:);
sknn_scores = sknn_scores(selected_labels,:);
svm_scores = svm_scores(selected_labels,:);
logistic_fairness = logistic_fairness(selected_labels,:);
sknn_fairness = sknn_fairness(selected_labels,:);
svm_fairness = svm_fairness(selected_labels,:);
logistic_auc = logistic_auc(selected_labels,:);
sknn_auc = sknn_auc(selected_labels,:);
svm_auc = svm_auc(selected_labels,:);

%%show new figure with Nemeyi rankings
% figure('Name', 'Improvement Tests' , 'NumberTitle', 'off', 'Position', get(0, 'ScreenSize'));
% clf
% subplot(2,3,1);
% friedman_test(-logistic_scores',labels,p,'Logistic wTPr.Fairness');
% subplot(2,3,2);
% friedman_test(-sknn_scores',labels,p,'Smooth-KNN wTPr.Fairness');
% subplot(2,3,3);
% friedman_test(-svm_scores',labels,p,'SVM wTPr.Fairness');
% subplot(2,3,4);
% friedman_test(-logistic_fairness',labels,p,'Logistic Fairness');
% subplot(2,3,5);
% friedman_test(-sknn_fairness',labels,p,'Smooth-KNN Fairness');
% subplot(2,3,6);
% friedman_test(-svm_fairness',labels,p,'SVM Fairness');


h = figure('Name', 'temp');
clf;
friedman_test(-logistic_scores',labels,p,'Logistic wTPr.Fairness');
print(strcat('image/scores_logistic',suffix),'-dpng')
clf;
friedman_test(-sknn_scores',labels,p,'Smooth-KNN wTPr.Fairness');
print(strcat('image/scores_sknn',suffix),'-dpng')
clf;
friedman_test(-svm_scores',labels,p,'SVM wTPr.Fairness');
print(strcat('image/scores_svm',suffix),'-dpng')
clf;
friedman_test(-logistic_fairness',labels,p,'Logistic Fairness');
print(strcat('image/fairness_logistic',suffix),'-dpng')
clf;
friedman_test(-sknn_fairness',labels,p,'Smooth-KNN Fairness');
print(strcat('image/fairness_sknn',suffix),'-dpng')
clf;
friedman_test(-svm_fairness',labels,p,'SVM Fairness');
print(strcat('image/fairness_svm',suffix),'-dpng')
clf;
friedman_test(-logistic_auc',labels,p,'Logistic wAUC');
print(strcat('image/wauc_logistic',suffix),'-dpng')
clf;
friedman_test(-sknn_auc',labels,p,'Smooth-KNN wAUC');
print(strcat('image/wauc_sknn',suffix),'-dpng')
clf;
friedman_test(-svm_auc',labels,p,'SVM wAUC');
print(strcat('image/wauc_svm',suffix),'-dpng')
clf;
close(h);