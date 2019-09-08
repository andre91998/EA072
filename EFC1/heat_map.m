%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Andr� Barros de Medeiros
%Date:09/05/2019
%Description: Display heatmaps for each of the 10 classes to help in the
%interpetability of the model
%Copyright: free to use, copy, and modify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n=1:1:10
    i=1;
    H=zeros(28,28);
    for j=1:1:784
        r=rem(j,28);
        if r==0
            i=i+1;
            r=1;
        end
        H(r,i)=W(j,n);

    end
    figure(n);
    heatmap(H,'GridVisible','off','Colormap',pink);
end
