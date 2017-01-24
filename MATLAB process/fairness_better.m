function fair = fairness_better(f, MCW)
    nom = 0;
    denom = 0;
    total_MCW = sum(f.*MCW);
    for i=1:length(f)
        local_fairness = MCW(i)*(1-f(i)) /  (total_MCW - MCW(i)*f(i));
        local_weight = 1/(f(i)*(1-f(i)));
        nom = nom + local_weight;
        denom = denom + local_weight*local_fairness;
    end
    fair = denom/nom;
end