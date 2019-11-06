%-------------------------------------------------------------------------%
%Displays the Kohonen map (one - dimensional)
%Parameters: X - Input data matrix - N (atributes) x M (paterns)
%            W - Weight matrix - N x neurons
%            neurons - number of neurons on the map
%-------------------------------------------------------------------------%

function plot_SOM(X, W, neuronios)
clf;
%We have all weights in W
plot(X(1,:), X(2,:),'r*'); hold on;
plot(W(1,:), W(2,:),'bo'); plot(W(1,:), W(2,:),'b');
%Connect first neuron to last
line([W(1,1) W(1,neuronios)],[W(2,1) W(2,neuronios)]);