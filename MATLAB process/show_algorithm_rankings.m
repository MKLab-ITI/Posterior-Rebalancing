function Ffriedman = show_algorithm_rankings(scores, labels, title, subtitle)
    if(nargin<4)
        subtitle = '';
    end
    
    %create figure
    %figure('Name', title, 'NumberTitle', 'off');
    %clf
    axis off
    axis([-0.2 1.2 60 140]);
    axis xy 
    %calculate average algorithm rankings
    algorithm_rankings = zeros(size(scores,1), 1);
    for j=1:size(scores,2)
        [~, algorithm_order] = sort(scores(:,j));
        for i=1:length(algorithm_order)
            min_equal = i;
            max_equal = i;
            for k=1:length(algorithm_order)
                if(scores(algorithm_order(i),j)==scores(algorithm_order(k),j))
                    min_equal = min(min_equal, k);
                    max_equal = max(max_equal, k);
                end
            end
            algorithm_rankings((algorithm_order(i))) = algorithm_rankings((algorithm_order(i))) + (min_equal+max_equal)/2;
        end
    end
    %normalize
    k = length(algorithm_rankings);%the number of algorithms
    N = size(scores,2);%the number of tests
    algorithm_rankings = algorithm_rankings / N;
    
    %calculate the square of Friedman statistic (follows a x^2 distribution)
    Xfriedman2 = 12*N/(k*(k-1))*sum(algorithm_rankings.^2-k*(k+1)*(k+1)/4);
    
    %derive Iman & Davenport statistic (follows F-distribution)
    Ffriedman = (N-1)*Xfriedman2/(N*(k-1)-Xfriedman2);
    
    %normalize rankings according to Nemenyi (this yields differences in
    %normal distribution significance)
    algorithm_rankings = algorithm_rankings*sqrt(k*(k+1)/(6*N));
    
    
    line([0 1], [100 100], 'LineWidth', 1, 'Color', 'k');
    text(0.5, 115, title, 'Color', 'blue', 'HorizontalAlignment', 'center')
    text(0.5, 92, subtitle, 'HorizontalAlignment', 'center')
    rating_fractions = algorithm_rankings/size(scores,1);
    rating_fractions = (rating_fractions-min(rating_fractions))/(max(rating_fractions)-min(rating_fractions));
    for i=1:length(algorithm_rankings)
        line_position = rating_fractions(i);
        line([line_position line_position], [103 97], 'LineWidth', 1, 'Color', 'k');
        text(line_position, 106, labels{i}, 'HorizontalAlignment', 'center');
        text(line_position, 95, num2str(round(algorithm_rankings(i)*100)/100), 'HorizontalAlignment', 'center');
    end
end