close all;clear;clc;
% solve (A+BC)x=b
N=2000;
k=100;
delta=1;
eta=1;

e=ones(N,1);

A=rand(k,N);
b=rand(k,1);
D=spdiags([-e 2*e -e],[-1 0 1],N,N);
D(1,1)=1;D(N,N)=1;
I=speye(N);


x=solver(delta*D+eta*I,A',A,A'*b);


max(abs((A'*A+delta*D+eta*I)*x-A'*b))