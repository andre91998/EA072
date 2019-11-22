i=0;
for j=1:0.1:10
    i=i+1;
    noise=10*rand(91,1);
    x1(1,i)=j;
    x2(1,i)=1/j;
    x3(1,i)=j^2;
    y(1,i) = 0.1*x1(1,i)-log(x2(1,i))+1/exp(x3(1,i));
    x1=transpose(x1);
    x2=transpose(x2);
    x3=transpose(x3);
    y=transpose(y);
end
