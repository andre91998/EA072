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
