clear;clc;
n=5;
e=ones(n,1);
A=spdiags([-e 2*e -e],[-1 0 1],n,n);
b=rand(n,1);

w=2/3;
D=diag(A)/w;
P=spdiags(D,[0],n,n);
xguess=rand(n,1);

t=30;
rs=zeros(t,1);
for i=1:t
    xguess=(b+(P-A)*xguess)./D;
    r=norm(b-A*xguess);
    rs(i)=r;
end
% semilogy(1:t,rs);

t=linspace(0,n+1,1000);
j=[1 n];
w=2/3;
eigM=@(w,t)1-w+w*cos(t*pi/(n+1));
hold on;
for w=[1,0.8 2/3 1/2,1/4]
    lambda=eigM(w,t);
    sample=eigM(w,j);
    plot(t,lambda,'-',j,sample,'r.');
end
legend('1','sm','4/5','sm','2/3','sm','1/2','sm','1/4','sm');

