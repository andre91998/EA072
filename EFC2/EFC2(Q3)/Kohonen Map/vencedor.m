%-------------------------------------------------------------------------%
%Returns the winning neuron's index
%padrao = column vector with input atributes
%W = matrix with each neuron's weight
%-------------------------------------------------------------------------%

function [indice, value] = vencedor(padrao,W)

%Set vector with the distance between the neurons and the input
d = dist(padrao',W);
%Determine the winning neuron's index
[value,indice] = min(d);
