%Calculates the Euclidean distance between two individuals
function dist = distancia(aux, aux1)
dist = sqrt(sum((aux-aux1).^2));
