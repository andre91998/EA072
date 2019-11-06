%-------------------------------------------------------------------------%
%Eliminate the neurons that never win
%Parameters: W - weight matrix (N x neurons)
%            wins - stores the victories (1 x neurons)
%            neurons - number of neurons on the map
%            Index - Neuron Neighborhood matrix
%-------------------------------------------------------------------------%

function [W,wins,neuronios,Index] = poda(W,wins,neuronios,Index)

%For all neurons on the map
te = neuronios;
while te >= 1
    %check if the neuron has never won
    if(wins(te) == 0)
        W(:,te) = []; %Delete neuron's column
        wins(:,te) = []; %Delete neuron's column
        neuronios = neuronios - 1; %Reduce the number of neurons on the map
        %Ajust Neighborhood
        Index(1,2)=neuronios; Index(neuronios,1)=1; Index(neuronios+1,:)=[];
    end
    te = te - 1;
end