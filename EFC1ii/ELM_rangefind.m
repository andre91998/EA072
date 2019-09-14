%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: André Barros de Medeiros
%Date:09/09/2019
%Description: Extreme Learning Machine (ELM), regfactor rangefinder, 500
%Neurons
%Copyright: free to use, copy, and modify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
load('data.mat'); %Dataset Matrix
n=0;
Neurons=500;
Mean=0;
StdDev=0.2;

%dividing X and S into training and validation
Xtrain=X(1:50000,:);
Xval=X(50001:60000,:);
Strain=S(1:50000,:);
Sval=S(50001:60000,:);

%Hidden Layer Matrix (H)
neuron_weights=StdDev*randn(784,500)+Mean;
Hi=radbas(Xtrain*(neuron_weights));
um=ones(50000,1);
H=[Hi um];

EQM=zeros(1,11); %Mean quadratic error matrix
TCC=zeros(1,11);%percentage of correct classifications for each regfactor
regfactorV=zeros(1,11); %regfactor storage vector

for i = 1:2:21
    certo=0;
    n=n+1;
    regfactor = 2^(i-5);%regulation factor
    regfactorV(n)=regfactor;%regulation factor vector (for plot)

    %calculate parameter matrix and classification matrix
        %Obs: N>=(n+1)
    W=(inv((H.')*H+(regfactor)*eye(501,501))*(H.')*Strain);
    Sc=([(Xval*neuron_weights) ones(10000,1)])*W;

    %calculate mean square error (EQM) with regfactor
    EQM(n)=(norm(Sc-Sval))^2+regfactor*(norm(W))^2;

    %evaluate correct classification percentage (TCC)
    for j = 1:1:10000
        Ic=zeros(1,1);
        I=zeros(1,1);
        [Mc,Ic]=max(Sc(j,1:10),[],'linear');
        [M,I]=max(Sval(j,1:10),[],'linear');
        if I(1)==Ic(1)
            certo=certo+1; %counting correct classifications variable
        end
    end
    TCC(n)=certo/10000;
end

%plot influence of regfactor on EQM and TCC
subplot(1,2,1); semilogx(regfactorV,EQM), xlabel('Coeficiente de Regularizacao'), ylabel('Erro Quadratico na Validacao'), grid;
subplot(1,2,2); semilogx(regfactorV,TCC), xlabel('Coeficiente de Regularizacao'), ylabel('Taxa de Classificacao Correta'), grid;