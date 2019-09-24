%-----------------------------------------------------------------------------%
%Author: André Barros de Medeiros
%Date:09/05/2019
%Copyright: free to use, copy, and modify
%Description: Linear Ridge Regression to classify MNIST dataset images
%Important: Training: 1:40000 (66% of dataset)
%           Validation: 40001:60000 (33% of dataset)
%-----------------------------------------------------------------------------%

load('data.mat'); %MNIST formated dataset provided
lambda=1; %linear regression control factor. Find math in PDF

%Initialize weight matrix (W) and Dataset matrix with one column (see PDF)
W=zeros(785,10,21);
column=ones(60000,1);
X=[column X];
lambdaV=zeros(1,21);%store all factors

%--------------------W for pre-defined values of lambda-----------------------%
i=0;
while i<21
    lambda=2^(i*2-14); %pre-defined lambdas in geometric progression, q=2
    lambdaV(i+1)=lambda;
    W(:,:,i+1)= ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(785))^-1)*X(1:40000,:)'*S(1:40000,:);%calculate W (see PDF)
    i=i+1;
end

miss= zeros(1,21);%miss classification count vector
hit= zeros(1,21);%correct classification count vector
hitpercentage=zeros(1,21); 
MeanSquareError=zeros(1,21);

j=1;

while j<=21
    miss(j)=0;
    hit(j)=0;
    
    i=40000+1;
    while i<=60000
        Scalc=X(i,:)*W(:,:,j); %Classify next validation example with W from above
        [~, indexMaxScalc] = max(Scalc); %get position of classification
        [~, indexMaxS] = max(S(i,:)); %get position of dataset answer

        if indexMaxS == indexMaxScalc %if classification is correct
            hit(j) = hit(j) + 1;
        else                          %if classification is incorrect
            miss(j) = miss(j) + 1;
        end

        MeanSquareError(j) = MeanSquareError(j) + mean((Scalc-S(i,:)).^2); %compute Square Error (see PDF)
        i=i+1;
    end

    MeanSquareError(j)=MeanSquareError(j)/20000; %divide by validation dataset size to have Mean SE
    hitpercentage(j)= hit(j)/(hit(j)+miss(j)); %compute percentage of correct classification
    j=j+1;
end

[bestMeanSquareError,bestMeanSquareErrorPos ] = min(MeanSquareError); %value and position of minimum MSE
[besthitpercentage,besthitpercentagePos ] = max(hitpercentage); %value and position of maximum correct classification

%Plot
figure(1);
subplot(1,2,1); semilogx(lambdaV,MeanSquareError), xlabel('Coeficiente de Regularizacao'), ylabel('Erro Quadratico Medio na Validacao'), grid;
subplot(1,2,2); semilogx(lambdaV,hitpercentage), xlabel('Coeficiente de Regularizacao'), ylabel('Taxa de Classificacao Correta'), grid;

%--------------------------------end Section------------------------------------%



%--------------W with refined Lambda Calculation for Mean Square Error----------%

W_refined = zeros(785, 10, 21);
lambdaV_refined=zeros(1,21);

j=0;
while j < 21
    i= bestMeanSquareErrorPos-2; %perform refined search with best previous lambda centralized
    lambda = 2^(i*2-14 + j*0.2); % multiply by 2^0.2 each iteration
    lambdaV_refined(j+1) = lambda;
    W_refined(:,:, j+1) = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(785))^-1)*X(1:40000,:)'*S(1:40000,:);
    j = j+1;
end

MeanSquareError=zeros(1,21);
j=1;

while j<=21    
    i=40000+1;
    while i<=60000
        Scalc=X(i,:)*W_refined(:,:,j); 
        MeanSquareError(j) = MeanSquareError(j) + mean((Scalc-S(i,:)).^2);
        i=i+1;
    end
    MeanSquareError(j)=MeanSquareError(j)/20000;
    j=j+1;
end

[bestMeanSquareError_Refined,bestMeanSquareErrorPos_Refined ] = min(MeanSquareError); %values not used but are usefull extra information
%Plot
figure(2);
semilogx(lambdaV_refined,MeanSquareError), xlabel('Coeficiente de Regularizacao'), ylabel('Erro Quadratico Medio na Validacao'), grid;
%-------------------------------End Section------------------------------------%



%-------------W with refined Lambda Calculation for hit percentage-------------%
W_refined = zeros(785, 10, 21);
lambdaV_refined=zeros(1,21);

j=0;
while j < 21
    i= besthitpercentagePos-2; %perform refined search with best previous lambda centralized
    lambda = 2^(i*2-14 + j*0.2);
    lambdaV_refined(j+1) = lambda;
    W_refined(:,:, j+1) = ((X(1:40000,:)'*X(1:40000,:)+lambda*eye(785))^-1)*X(1:40000,:)'*S(1:40000,:);  
    j = j+1;
end

miss= zeros(1,21);
hit= zeros(1,21);
hitpercentage=zeros(1,21);

j=1;

while j<=21
    miss(j)=0;
    hit(j)=0;
    i=40000+1;
    while i<=60000
        Scalc=X(i,:)*W_refined(:,:,j); 
        
        [~, indexMaxScalc] = max(Scalc);
        [~, indexMaxS] = max(S(i,:));

        if indexMaxS == indexMaxScalc
            hit(j) = hit(j) + 1;
        else
            miss(j) = miss(j) + 1;
        end        
        i=i+1;
    end

    hitpercentage(j)= hit(j)/(hit(j)+miss(j));
    j=j+1;
end

[besthitpercentage_Refined,besthitpercentagePos_Refined ] = min(MeanSquareError);

figure(3);
semilogx(lambdaV_refined,hitpercentage), xlabel('Coeficiente de Regularizacao'), ylabel('Hit Percentage'), grid;

%Since correct classification is a much better criterion than MSE for a
%linear regression, store W_refined in this case as our final W (which will
%be used to calculate confusion matrix and heatmaps (and used do classify future data)
W_final=W_refined(:,:,besthitpercentagePos_Refined); 
lambda_final = lambdaV_refined(besthitpercentagePos_Refined);

%----------------------------------End Section----------------------------------%



%----------------------- Model Analysis and Interpretability--------------------%

%---Calculate Confusion Matrix---%
ConfM=zeros(10,10); %initialize confusion matrix
e=1;
Scalc=X(40001:60000,:)*W_final(:,:); %classify validation dataset
for j = 1:1:20000
    Ic=zeros(1,1);
    I=zeros(1,1);
    [Mc,Ic]=max(Scalc(j,1:10),[],'linear'); %value and position of max value in row "j" (classification)
    [M,I]=max(S(j+40000,1:10),[],'linear'); %value and position of max value in row "j" (correct answer)
    if I(1)==Ic(1)
        ConfM(I,I)=ConfM(I,I)+1; %if model classified correctly, ++ in correct position of matrix diagonal
    end
    if I(1)~=Ic(1)               %if model classified incorrectly, ++ in position I,Ic
        ConfM(I,Ic)=ConfM(I,Ic)+1;
        if e<=4               
            Wrong(e)=40000+j;   %store postion in dataset of 4 wrongly classified images
            e=e+1;
        end
    end
end

ConfM;

%---Generate Images of wrongly classified data---%
for n=1:1:4
    i=1;
    H=zeros(24,24); %matrix to be printed
    for j=1:1:784   %loop takes dataset row and re-makes image matrix
        r=rem(j,28);
        if r==0
            i=i+1;
            r=1;
        end
        H(r,i)=X(Wrong(n),j);
    end
    [Me,Ie]=max(S((Wrong(n)),1:10),[],'linear'); %get the correct class of wrongly classified image (Ie)
    if Ie==10     %if Ie=10, class is number 0
        Ie=0;
    end
    figure(n+4);
    heatmap(H,'GridVisible','off','Colormap',gray,'Title', Ie);
end

%---HeatMap---%
for n=1:1:10
    i=1;
    H=zeros(28,28);
    for j=1:1:784 %same re-make image for loop as above
        r=rem(j,28);
        if r==0
            i=i+1;
            r=1;
        end
        H(r,i)=W_final(j,n); %here we make "image" with weight matrix. One image for each column W (different classes)
    end
    if n<10
        title=n;
    else
        title=0;
    end
    figure(n+8);
    heatmap(H,'GridVisible','off','Colormap',parula,'Title', title);
end

%----------------------------------End Section---------------------------------%

