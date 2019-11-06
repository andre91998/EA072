%-------------------------------------------------------------------------%
%Kohonen Map training - One-dimensional disposition - circle
%Automatic removal and insertion of neurons
%Matrix X: (N atributes x M patterns)
%   Collumns: represent input patterns
%-------------------------------------------------------------------------%

function [W,Index,neuronios] = kohonen(X,neuronios,gama_inicial,radius_inicial,LIMIAR,LIMITE_TAXA,PERIODO,MAX_ITERATION)

[sem_uso,M] = size(X);

%Number of Iterations
iteration = 1;
%Learning Rate
gama = gama_inicial; 
%Radius defines neighborhood - initially, both neighbors are affected
%As the number of iterations increases, ajustment no longer happens
radius = radius_inicial;

%Matrix with topological neighbor indices - circle
vetor = (1:neuronios)';
Index = [circshift(vetor,-1) circshift(vetor,1)];
%Vector that counts neuron victories
wins = zeros(1,neuronios);

%Initializes Weight Matrix
W = inicializa_pesos(X,neuronios);
%Display initial configuration - weights and data points
figure(1); plot_SOM(X,W,neuronios);
title('Initial Neuron Configuration');

%Counts the number of neurons close to the patterns
cont = 0; 

%Stop Criteria:
%   Maximum number of epochs
%   Close to pattern neuron counter: means that distante to closest data 
%       point (city) is bellow THRESHOLD
%   Learning rate above the permitted level

while iteration < MAX_ITERATION && cont < M && gama > LIMITE_TAXA
    
    %Each Iteration set victory counter to zero
    if(mod(iteration,PERIODO)==0)
        wins = zeros(1,neuronios);
    end
    
    %Randomize pattern input order
    X = X(:,randperm(M));
    %Counts the number of neurons close to the patterns
    cont = 0;
   
    %Displays the M patterns
    for i=1:1:M
        %Determin winning neuron
        [indice, value] = vencedor(X(:,i),W);
        %Add a victory to the neuron "indice"
        wins(indice) = wins(indice) + 1;
        if value < LIMIAR %if all the neurons are closer to data points than the THRESHOLD
           cont = cont + 1;
        end
        %Ajust the winning neuron weihgt and that of its neighbors
        W = ajuste_peso(W, X, Index, indice, gama, i, radius);
    end
    
    %Ajust learning rate "gama"and the neighborhood radius
    gama = gama_inicial*exp(-(iteration)/MAX_ITERATION);  %time constant
    radius = radius_inicial*exp(-(iteration)/MAX_ITERATION);
    
    %Each Iteration, eliminate and insert neurons
    if(mod(iteration,PERIODO)==0)
        %Elimination - Neurons that never win
        [W,wins,neuronios,Index] = poda(W,wins,neuronios,Index);
        %Insertion - Near neurons that win a lot
        [W,wins,neuronios,Index] = insere(W,wins,neuronios,Index);
    end
    %Display Parameters
    fprintf('Iteration:%d \t Rate (Gama):%1.4f \t Radius:%d \t N:%d\n',iteration,gama,round(radius),neuronios);
    figure(2); plot_SOM(X,W,neuronios); title(['Neuron Configuration - iteration ' num2str(iteration)]); drawnow;
    iteration = iteration + 1;
end