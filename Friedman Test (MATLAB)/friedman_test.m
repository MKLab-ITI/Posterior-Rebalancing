function [ranks, avg_ranks, result] = friedman_test(data, labels, alpha, fig_title)
%FRIEDMAN_TEST returns the result, significant differences, average ranks, critical difference and ranks for each data set using Friedman's test
%   result is 0 if there is no significant difference between the algorithms, 1 o/w
%   sig_diffs is a matrix of 0s and 1s of size (num_algorithms, num_algorithms) and an entry (i, j) is 1 if the accuracies of algorithm i and algorithm j are significantly different from each other.
%   avg_ranks is the average ranks of the algorithms
%   CD is the critical distance to separate algorithms. Makes sense for Nemenyi's test
%   ranks is the ranks of all data sets
%   data should be of size (num_algorithms, num_datasets, num_folds) and contain errors or ranks
%   alpha should be given as 0.01, 0.001, 0.05 (not 0.99 etc)
%   posthoc_test should be one of 'bergman', 'nemenyi', 'holm', 'shaffer'
%   Aydin Ulas, Mehmet Gonen, Department of Computer Engineering, Bogazici University
%   $Revision: 1.00 $  $Date: 2009/05/12 $
 
data = data';
     
num_algorithms = size(data, 1);
if num_algorithms > 50
    display('classifier count should be less than or equal to 50!!!');
    return;
end
num_datasets = size(data, 2);
num_folds = size(data, 3);
for i = 1:num_algorithms
    performance(:, i) = mean(reshape(data(i, :, :), num_datasets, num_folds), 2);
end
for i = 1:num_datasets
    ranks(i, :) = tiedrank(performance(i,:));
end
ranks;
avg_ranks = mean(ranks, 1);
chi_statistic = (12 * num_datasets) * (sum(avg_ranks.^2) - 0.25 * num_algorithms * (num_algorithms + 1)^2)/ (num_algorithms * (num_algorithms + 1));
f_statistic = ((num_datasets - 1) * chi_statistic)/ (num_datasets * (num_algorithms-1) - chi_statistic);
chi_critical = chi2inv(1 - alpha, num_algorithms - 1);
f_critical = finv(1 - alpha, num_algorithms - 1, (num_algorithms - 1)*(num_datasets-1));
 
v1=num_algorithms - 1;
v2=(num_algorithms - 1)*(num_datasets-1);
p_critical = fcdf(f_critical,v1,v2,'upper')
p_actual = fcdf(f_statistic,v1,v2,'upper')
 
if abs(chi_statistic) > chi_critical && abs(f_statistic) > f_critical 
    result = 'We reject the null hypothesis with both statistics.'
    cd = criticaldifference(data',labels,alpha,fig_title,p_actual);
elseif abs(f_statistic) > f_critical 
    result = 'We reject the null hypothesis only with f statistic.'
    cd = criticaldifference(data',labels,alpha,fig_title,p_actual);
else
    result = 0
    cd = criticaldifference(data',labels,alpha,fig_title,p_actual);
end