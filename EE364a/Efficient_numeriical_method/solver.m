function x = solver(A,B,C,b)
% solve (A+BC)x=b

% z1=A\b Z1=A\B
k=size(B,2);
I=speye(k);

Z=A\[b B];
% y=C*z1 H=C*Z1
T=C*Z;
% (I+H)\y
y=(I+T(:,2:end))\T(:,1);
x=Z(:,1)-Z(:,2:end)*y;


end

