%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: André Barros de Medeiros
%Date:09/05/2019
%Description: Compute Confusion Matrix to make model more interpretable
%Copyright: free to use, copy, and modify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
load('data.mat');%Data Matrix

%Parameter Matrix
W=(-1 + 2*rand(785,10))/10;
W(:,1)=1;

%dividing X and S into training and validation
Xtrain=X(1:40000,:);
Xval=X(40001:60000,:);
Strain=S(1:40000,:);
Sval=S(40001:60000,:);

regfactor = 1007; %determined with regfactor_refined.m
Wrong=zeros(1,4);%store positions of 4 wrongly classified images
e=1; %counting variable for wrongly classified images

%calculate parameter matrix and classification matrix
W=(inv((Xtrain.')*Xtrain+(regfactor)*(eye(784,784)))*(Xtrain.')*Strain);
Sc=Xval*W;

%Compute Confusion Matrix%
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
        if e<=4
            Wrong(e)=40000+j;
            e=e+1;
        end
    end
end

ConfM;

%Generate Images of wrongly classified images
for n=1:1:4
    i=1;
    H=zeros(24,24);
    for j=1:1:784
        r=rem(j,28);
        if r==0
            i=i+1;
            r=1;
        end
        H(r,i)=X(Wrong(n),j);
    end
    [Me,Ie]=max(S((Wrong(n)),1:10),[],'linear');
    figure(n);
    heatmap(H,'GridVisible','off','Colormap',gray,'Title', Ie);
end
