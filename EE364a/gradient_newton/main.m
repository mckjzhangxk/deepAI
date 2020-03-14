clc;clear;
%probelem setup up
randn('state',1);
m=200;n=100;
A=randn(m,n);
f=@(y) fobj(A,y); 
df=@(y) fobj_grad(A,y);
d2f=@(y) fobj_hessian(A,y);

gradDecendMethod=@(x) -df(x);
NewtonMethod=@(x) -d2f(x)\df(x);

%solver hyparams
ALPHA=0.01;
BETA=0.5;
x=zeros(n,1);
eps=1e-3;



history=[];
ts=[];
% methodName='gradient decent';
% method=gradDecendMethod;
methodName='Newton';
method=NewtonMethod;

while true
    g=fobj_grad(A,x);
    %find good direction 
    dx=method(x);
    %find best step size
    t=linear_search(f,x,dx,g,ALPHA,BETA);
    %update x
    x=x+t*dx;
    
    decrement=sum(-dx.*g);
    if decrement<eps
        break;
    end
    
    history=[history f(x)];
    ts=[ts t];
end
p_opval=f(x);
figure();
semilogy(1:length(history),history-p_opval);
xlabel('iteration(n)');
ylabel('f(x)-p*');
title(methodName);

figure();
plot(1:length(ts),ts,'o');
xlabel('iteration(n)');
ylabel('step size');
title(methodName);