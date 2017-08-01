function fair = fairness(f, MCW)
    nom = 0;
    denom = 0;
    for i=1:length(f)
        nom = nom + f(i)*(1-f(i));
        for j=1:length(f)
            if(i~=j)
                denom = denom + f(i)*f(j)*MCW(i)/MCW(j);
            end
        end
    end
    fair = nom/denom;
end