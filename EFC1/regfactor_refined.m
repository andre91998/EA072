%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Andr� Barros de Medeiros
%Date:09/05/2019
%Description: Find optimal regulation factor with a refined search in
%pre-established range (regfactor =1007)
%Copyright: free to use, copy, and modify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
load('data.mat'); %Dataset Matrix
n=0;

%Parameter Matrix
W=(-1 + 2*rand(785,10))/10;
W(:,1)=1;

EQM=zeros(1,101); %Mean quadratic error (EQM)
TCC=zeros(1,101); %percentage of correct classifications for each regfactor(TCC)
regfactorV=zeros(1,101);  %regfactor storage vector

for i = 1000:1:1100
    certo=0;
    n=n+1;
    regfactor = i;
    regfactorV(n)=regfactor;
    
    %dividing X and S into training and validation
    Xtrain=X(1:40000,:);
    Xval=X(40001:60000,:);
    Strain=S(1:40000,:);
    Sval=S(40001:60000,:);
    
    %calculate parameter matrix and classification matrix
    W=(inv((Xtrain.')*Xtrain+(regfactor)*(eye(784,784)))*(Xtrain.')*Strain);
    Sc=Xval*W;
    
    %calculate mean square error
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

%plot influence of refined regfactor on EQM and TCC
subplot(1,2,1); semilogx(regfactorV,EQM), xlabel('Coeficiente de Regularizacao'), ylabel('Erro Quadratico na Validacao'), grid;
subplot(1,2,2); semilogx(regfactorV,TCC), xlabel('Coeficiente de Regularizacao'), ylabel('Taxa de Classificacao Correta'), grid;

%Compute Confusion Matrix%
% ConfM=zeros(10,10);
% for j = 1:1:20000
%     Ic=zeros(1,1);
%     I=zeros(1,1);
%     [Mc,Ic]=max(Sc(j,1:10),[],'linear');
%     [M,I]=max(Sval(j,1:10),[],'linear');
%     if I(1)==Ic(1)
%         ConfM(I,I)=ConfM(I,I)+1;
%     end
%     if I(1)~=Ic(1)
%         ConfM(I,Ic)=ConfM(I,Ic)+1;
%     end
% end
    
