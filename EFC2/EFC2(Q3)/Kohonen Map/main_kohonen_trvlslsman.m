%-------------------------------------------------------------------------%
%Author: André Barros de Medeiros
%Date: 11/06/19
%Copyright: Free to use, copy, and modify
%Description:
%   Toolbox - One-dimensional Kohonen Map applied to the Travelling 
%             Salesman Problem
%   Specifications:
%       Neighborhood - one-dimensional in in a circle
%       Number of Neurons - controlled dynamically
%-------------------------------------------------------------------------%

clear; close all;

%Load coordinates of all the cities
load dados.mat

%Selects an instance of the Travelling Salesman
choice = menu('Choose which map you want:','Berlin52','Inst. 1 (100 cidades) ','Inst. 2 (100 cidades)');
if choice == 1
    load berlin52
elseif choice == 2
    load inst100x100.mat
else
    load dados.mat;
end

%Self Organizing Map (SOM) parameters

%Initial Number of Neurons
N = 10;
%Maximum number of epochs in the auto-organization process
max_epoch = 500;
%Number of epochs for victory count to reach zero and to perform Neuron number control
PERIODO = 5;
%Winning Neuron proximity to pattern threshold
limiar = 0.01;
%Initial learning rate
gama = 0.2;
%Learning rate threshold (minimum value permitted)
limite_taxa = 0.01;
%Neighborhood radius - in the beginning, the neighbors (right and left) are more intensely affected; As the number of epochs grows, the ajustment decreases
radius = 1;

%Kohonen SOM
[W,Index,Nf] = kohonen(X,N,gama,radius,limiar,limite_taxa,PERIODO,max_epoch);

%Display Results
close(2); figure; plot_SOM(X,W,Nf);

%Determin the path to be taken

%Matrix with distances between each neuron and city
mt = dist(W',X);
%Matrix with distances between cities
mx = dist(X', X); 
solucao = zeros(1,Nf); d = 0;
%Find Departure City (first city)
[sem_uso,ind] = min(mt(1,:));
solucao(1) = ind;

for k=1:Nf-1
    [sem_uso,ind] = min(mt(k+1,:)); %ind = index of the city represented by the Kth neuron
    %montamos o percurso com a ordem das cidades
    solucao(k+1) = ind;
    %Add up distance between cities (get total distance)
    d = d + mx(solucao(k),solucao(k+1));
end
%Include also the distance between the first and last city
d = d + mx(solucao(1),solucao(Nf));
title(['Final Neuron Configuration - Complete Route: ' num2str(d)]);
