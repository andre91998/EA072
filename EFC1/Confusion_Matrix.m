%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Andr� Barros de Medeiros
%Date:09/05/2019
%Description: 1007
%Copyright: free to use, copy, and modify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
%Data Matrix
load('data.mat');
%Parameter Matrix
W=(-1 + 2*rand(785,10))/10;
W(:,1)=1;

regfactor = 1007; %determined with regfactor_refined.m
%dividing X and S into training and validation
Xtrain=X(1:40000,:);
Xval=X(40001:60000,:);
Strain=S(1:40000,:);
Sval=S(40001:60000,:);
%calculate parameter matrix
W=(inv((Xtrain.')*Xtrain+(regfactor)*(eye(784,784)))*(Xtrain.')*Strain);
Sc=Xval*W;

%Confusion Matrix%
ConfM=zeros(10,10);
for j = 1:1:20000
    Ic=zeros(1,1);
    I=zeros(1,1);
    [Mc,Ic]=max(Sc(j,1:10),[],'linear');
    [M,I]=max(Sval(j,1:10),[],'linear');
    if I(1)==Ic(1)
        ConfM(I,I)=ConfM(I,I)+1;
    end
    if I(1)~=Ic(1)
        ConfM(I,Ic)=ConfM(I,Ic)+1;
    end
end

ConfM;
