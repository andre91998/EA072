i=0;
for j=1:0.1:10
    i=i+1;
    noise=0.2*rand(91,1);
    x1(1,i)=j+5;
    x2(1,i)=-j;
    x3(1,i)=(j-4.5)^2;
    y(1,i) = x1(1,i)+cos(x2(1,i))+sin(x3(1,i))+noise(i,1);
    x1=transpose(x1);
    x2=transpose(x2);
    x3=transpose(x3);
    y=transpose(y);
end
