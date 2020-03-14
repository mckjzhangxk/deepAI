clc;clear;
team_data;
A=sparse(1:m,train(:,1),train(:,3),m,n)+sparse(1:m,train(:,2),-train(:,3),m,n);

cvx_begin
    variable a(n);
    minimize(-sum(log_normcdf(A*a/sigma)))
    subject to
        0<=a<=1;
cvx_end

y=test(:,3);
ypredict=sign(a(test(:,1))-a(test(:,2)));
acc=sum((y==ypredict))/size(y,1);