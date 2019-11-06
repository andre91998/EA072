%Ajusts winning neuron weight and that of its neighbors
function W = ajuste_peso(W, X, Index, indice, gama, i_padrao, radius)

%Auxilary variables
padrao = X(:,i_padrao); %Vector with presented pattern
ind_sup = Index(indice,2); %Above neighbor index
ind_inf = Index(indice,1); %Below neighbor index
j = round(radius); %Simulates the neighborhood reduction
if j > 0
    %Ajust the "Above Neighbor" weight vector
    W(:,ind_sup) = W(:,ind_sup) + gaussmf(indice + 1,[radius,indice])*gama*(padrao - W(:,ind_sup));
    %Ajust the "Below Neighbor" weight vector
    W(:,ind_inf) = W(:,ind_inf) + gaussmf(indice - 1,[radius,indice])*gama*(padrao - W(:,ind_inf));
end
%Ajust the Winning Neuron weight vector
W(:,indice) = W(:,indice) + gama*(X(:,i_padrao) - W(:,indice));