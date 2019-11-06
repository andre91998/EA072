%-------------------------------------------------------------------------%
%Add neurons close to the winning neuron
%Parameters: W - weight matrix (N x neurons)
%            wins - vector that stores the victories (1 x neuronios)
%            neuronios - number of neurons on the map
%            Index - neighborhood matrix
%-------------------------------------------------------------------------%

function [W,wins,neuronios,Index] = insere(W,wins,neuronios,Index)

%Maximum number of victories
max_wins = max(wins);
%Percentage of the maximum number of victories
alpha = 0.5; %Controls the number o neurons that are inserted
te = 1;
%For all the neurons on the map
while te <= neuronios
    %If the neuron has reached the victory threshold
    if wins(te) > 1 && wins(te) >= alpha*max_wins
        %add a column to the victory vector
        wins = [wins(:,1:te) (wins(:,te)-1) wins(:,te+1:neuronios)];
        %add a column to the weight matrix
        W = [W(:,1:te) (0.02*rand-0.01+W(:,te)) W(:,te+1:neuronios)];
        %add a neuron
        neuronios = neuronios + 1;
        %increase the index as to not alter the neuron just created
        te = te + 1;
        %Ajust Neighborhood
        Index(1,2)=neuronios; Index(neuronios-1,1)=neuronios; Index(neuronios,:)=[1 neuronios-1];
    end
    %Next neuron(after the on just created)
    te = te + 1;
end