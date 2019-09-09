%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: André Barros de Medeiros
%Date:09/05/2019
%Description: Find range to search for the optimal regulation factor
%Copyright: free to use, copy, and modify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
load('data.mat'); %Dataset Matrix
n=0;

%Parameter Matrix
W=(-1 + 2*rand(785,10))/10;
W(:,1)=1;

EQM=zeros(1,11); %Mean quadratic error matrix
TCC=zeros(1,11);%percentage of correct classifications for each regfactor
regfactorV=zeros(1,11); %regfactor storage vector

for i = 1:2:21
    certo=0;
    n=n+1;
    
    regfactor = 2^(i-8);%regulation factor
    regfactorV(n)=regfactor;%regulation factor vector (for plot)
    
    %dividing X and S into training and validation
    Xtrain=X(1:40000,:);
    Xval=X(40001:60000,:);
    Strain=S(1:40000,:);
    Sval=S(40001:60000,:);
    
    %calculate parameter matrix and classification matrix
    W=(inv((Xtrain.')*Xtrain+(regfactor)*(eye(784,784)))*(Xtrain.')*Strain);
    Sc=Xval*W;
    
    %calculate mean square error (EQM)
    EQM(n)=(norm(Sc-Sval))^2+regfactor*(norm(W))^2;
    
    %evaluate correct classification percentage (TCC)
    for j = 1:1:20000
       Ic=zeros(1,1);
       I=zeros(1,1);
       [Mc,Ic]=max(Sc(j,1:10),[],'linear');
       [M,I]=max(Sval(j,1:10),[],'linear');
      if I(1)==Ic(1)
           certo=certo+1; %counting correct classifications variable
     end
    end
    TCC(n)=certo/20000;
end

%plot influence of regfactor on EQM and TCC
subplot(1,2,1); semilogx(regfactorV,EQM), xlabel('Coeficiente de Regularizacao'), ylabel('Erro Quadratico na Validacao'), grid;
subplot(1,2,2); semilogx(regfactorV,TCC), xlabel('Coeficiente de Regularizacao'), ylabel('Taxa de Classificacao Correta'), grid;
