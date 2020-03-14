clc;clear;


%probelem setup up
randn('state',1);
m=200;n=100;
A=randn(m,n);
f=@(y) fobj(A,y); 
df=@(y) fobj_grad(A,y);
d2f=@(y) fobj_hessian(A,y);
d2fd=@(y) fobj_diag_hessian(A,y);


%solver hyparams
ALPHA=0.01;
BETA=0.5;
x=zeros(n,1);
eps=1e-3;
MAXIters=1000;


% Nowtown method
% method=@(y,s)NewtonMethod(y,df,d2f);
% [x,history,ts]=solver(f,x,method,ALPHA,BETA,eps,MAXIters); 
% p_opval=f(x);
% semilogy(1:length(history),history-p_opval);


% reusing Hessian method
N=1;
x=zeros(n,1);
method=@(y,s)ReusingHessianMethod(y,df,d2f,s,N);
[x,history,ts]=solver(f,x,method,ALPHA,BETA,eps,MAXIters); 
p_opval=f(x);
semilogy(1:length(history),history-p_opval);


% diagonnal approximation
x=zeros(n,1);
method=@(y,s) DiagHessionMethod(y,df,d2fd);
[x,history,ts]=solver(f,x,method,ALPHA,BETA,eps,MAXIters); 
semilogy(1:length(history),history-p_opval);
xlabel('iteration(n)');
ylabel('f(x)-p*');

