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
Hi=tanh(Xtrain*(neuron_weights));
um=ones(50000,1);
H=[Hi um];

EQM=zeros(1,11); %Mean quadratic error matrix
TCC=zeros(1,11);%percentage of correct classifications for each regfactor

regfactor=1684;
e=1;
Wrong=zeros(1,4);
%calculate parameter matrix and classification matrix
    %Obs: N>=(n+1)
W=(inv((H.')*H+(regfactor)*eye(501,501))*(H.')*Strain);
Sc=([(Xval*neuron_weights) ones(10000,1)])*W;

%Compute Confusion Matrix%
ConfM=zeros(10,10);

%evaluate correct classification percentage (TCC)
for j = 1:1:10000
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
    if e<=4
        Wrong(e)=40000+j;
        e=e+1;
    end
end
    

ConfM;

%Generate Images of wrongly classified images
for n=1:1:4
    i=1;
    CM=zeros(24,24);
    for j=1:1:784
        r=rem(j,28);
        if r==0
            i=i+1;
            r=1;
        end
        CM(r,i)=X(Wrong(n),j);
    end
    [Me,Ie]=max(S((Wrong(n)),1:10),[],'linear');
    if Ie==10
        Ie=0;
    end
    figure(n);
    heatmap(CM,'GridVisible','off','Colormap',gray,'Title', Ie);
end