%-------------------------------------------------------------------------%
%Initializes weights - One-dimensional, circular disposition
%Parameters: X - Input data Matrix (NxM)
%
%This initialization aims to minimize topological violation in the 
%       construction of the map which means that after the self 
%       organization of the map, we want the neighboring neurons to have 
%       weight vectors close to one another in the "Data Space"
%-------------------------------------------------------------------------%

function W = inicializa_pesos(X, neuronios)

%N = number of components and M = number of input data points (patterns)
[N,M] = size(X);
%Mean data point (center of circle)
medio = mean(X,2)'; 
%Input data limits
limite_max=max(max(X));
limite_min=min(min(X));
%Guarantee mean=0 for input data
Xmz = X - repmat(medio',1,M);
%Autocorrelation matrix - PCA
R = Xmz*Xmz';
%Eigenvalues and Eigenvectors
[V,sem_uso] = eig(R);
%Two largest Eigenvalues define the two main components
direcao = [V(:,N-1) V(:,N)];
%circle radius
raio = abs(0.2*(limite_max - limite_min));
%Angle of variation
angles = linspace(0,2*pi,neuronios);
%Build a matrix with the neuron weights (a circle in the "Data Space")
W = cos(angles)'*raio*direcao(:,1)'+sin(angles)'*raio*direcao(:,2)'+repmat(medio',1,neuronios)';
W = W'; %Maintain manter coherence/consistency from one execution to the next
