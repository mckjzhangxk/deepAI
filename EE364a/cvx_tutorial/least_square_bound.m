clear;clc;
m=10;n=4;

A=randn(m,n);
b=randn(m,1);

bd=rand(m,2);
ll=0;
uu=max(bd,[],2)*3;

cvx_begin
    variable x(n);
    %norm(a,1),norm(a,inf)
    minimize(norm(A*x-b,2));
    subject to
        A*x>=ll;
        A*x<=uu;
cvx_end