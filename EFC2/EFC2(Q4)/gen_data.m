i=0;
for j=1:0.1:10
    i=i+1;
    noise=0.5*rand(91,1);
    x(1,i)=j;
    y(1,i)=j*cos(5.3+6*j)+noise(i);%+j*sin(j^2);
    x=transpose(x);
    y=transpose(y);
    %x=x(:,1)
    %y=y(:,1)
end
