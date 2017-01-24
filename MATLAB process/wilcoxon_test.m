function confidence=wilcoxon_test(values)
% WILCOXON_TEST Performs a wilcoxon sign test to determine the confidence
%               of a distribution difference being positive.
%
%   CONFIDENCE = WILCOXON_TEST(VALUES) where VALUES is the difference of
%               two distribition values on the same subjects (they need not
%               be ordered)
    
    %remove zero values
    values = values(values~=0);
    %sort from lesser to larger abs values (to do so, first obtain indexes)
    [~, sortedAbsValueIndex] = sort(abs(values));
    values = values(sortedAbsValueIndex);
    %obtain number of samples
    N = length(values);
    ranks = 1:N;
    %obtain W
    W = sum(ranks.*sign(values));
    %calculate variance of W distribution
    varW = sqrt(N*(N+1)*(2*N+1)/6);
    %obtain z value for W distribution (null hypothesis assumes mean(W)=0
    zValue = (W-0.5)/varW;
    %obtain confidence lever from normal distribution
    confidence = normcdf(zValue);
    %if the classifiers are the same, then confidence should be 0.5
    if(isempty(values))
        confidence = 0.5;
    end
end