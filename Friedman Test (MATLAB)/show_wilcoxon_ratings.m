function ratings = show_wilcoxon_ratings(scores, labels, title, subtitle, range)
% SHOW_WILCOXON_RATINGS Applies the wilcoxon sign test to compare various
%   classifier performance to a base one and then produces a graph that
%   visually compares them. Values mark the confidence of improvement.
%
%   SHOW_WILCOXON_RATINGS(SCORES, LABELS, TITLE) where SCORES is a matrix
%       whose i-th row contains the scores for the i-th classifier, LABELS
%       are the names of classifiers to be displayed and TITLE is the
%       figure title. The first classifier. All classifiers are compared to
%       the first classifier.
%
%   SHOW_WILCOXON_RATINGS(SCORES, LABELS, TITLE, SUBTITLE) also adds a
%       SUBTITLE text bellow the visualization (it''s empty by default).
%
%   SHOW_WILCOXON_RATINGS(SCORES, LABELS, TITLE, SUBTITLE, RANGE)
%       also scales scores to lie in the desired RANGE. Default value for
%       RANGE is [0 1].
%
%   See also; WILCOXON_TEST

    if(nargin<4)
        subtitle = '';
    end
    if(nargin<5)
        range = [0 1];
    end
    
    %create figure
    %figure('Name', title, 'NumberTitle', 'off');
    axis off
    axis([-0.2 1.2 60 140]);
    axis xy 
    %calculte wilcoxon test scores
    base = scores(1,:);
    wilcoxon_scores = zeros(size(scores,1), 1);
    for i=1:size(scores,1)
        wilcoxon_scores(i) = (1-signrank(scores(i,:)-base))*sign(mean(scores(i,:)-base));
    end
    wilcoxon_scores = wilcoxon_scores*(range(2)-range(1))+range(1);
    ratings = wilcoxon_scores;
    [wilcoxon_scores, wilcoxon_scores_index] = sort(wilcoxon_scores);
    
    wilcoxon_scores_range = wilcoxon_scores(length(wilcoxon_scores))-wilcoxon_scores(1);
    line([0 1], [100 100], 'LineWidth', 1, 'Color', 'k');
    text(0.5, 115, title, 'Color', 'blue', 'HorizontalAlignment', 'center')
    text(0.5, 92, subtitle, 'HorizontalAlignment', 'center')
    for i=1:length(wilcoxon_scores)
        line_position = (wilcoxon_scores(i)-wilcoxon_scores(1))/wilcoxon_scores_range;
        line([line_position line_position], [103 97], 'LineWidth', 1, 'Color', 'k');
        text(line_position, 106, labels{wilcoxon_scores_index(i)}, 'HorizontalAlignment', 'center');
        text(line_position, 95, strcat(num2str(round(wilcoxon_scores(i)*100)), '%'), 'HorizontalAlignment', 'center');
    end
    line_position = (0.95-wilcoxon_scores(1))/wilcoxon_scores_range;
    line([line_position line_position], [106 94], 'LineWidth', 2, 'Color', 'blue');
end