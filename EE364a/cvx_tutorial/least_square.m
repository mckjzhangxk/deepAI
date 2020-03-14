m=10;
n=4;
A=rand(m,n);
b=rand(m,1);

cvx_begin
    variable x(n);
    minimize(norm(A*x-b))
cvx_end