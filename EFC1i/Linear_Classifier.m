%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Andr� Barros de Medeiros
%Date:09/05/2019
%Description:
%Copyright: free to use, copy, and modify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
%Data Matrix
load('data.mat');
%Parameter Matrix
W=(-1 + 2*rand(785,10))/10;
W(:,1)=1;
%Mean quadratic error
EQM=zeros(1,23);
%counting correct classifications variable
%certo=0;
%percentage of correct classifications for each regfactor
TCC=zeros(1,23);

regfactorV=zeros(1,23);

for i = 1:1:23
    certo=0;
    regfactor = 2^(i-11);
    regfactorV(i)=regfactor;
    Xtrain=X(1:40000,:);
    Xval=X(40001:60000,:);
    Strain=S(1:40000,:);
    Sval=S(40001:60000,:);
    %calculate parameter matrix
    W=(inv((Xtrain.')*Xtrain+(regfactor)*(eye(784,784)))*(Xtrain.')*Strain);
    Sc=Xval*W;
    %EQM(i)=(norm(Sc-Sval))^2+regfactor*(norm(W))^2;
     
       for j = 1:1:20000
          Ic=zeros(1,1);
          I=zeros(1,1);
          [Mc,Ic]=max(Sc(j,1:10),[],'linear');
          [M,I]=max(Sval(j,1:10),[],'linear');
         if I(1)==Ic(1)
              certo=certo+1;
         end
       end
       TCC(i)=certo/20000;
end

semilogx(regfactorV,TCC);
