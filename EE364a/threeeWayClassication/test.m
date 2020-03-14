clc;clear;
X=randn(10,2)';
Y=randn(10,2)';
Z=randn(10,2)';


cvx_begin
    variables a1(2) a2(2) a3(2) b1 b2 b3 L1 L2 L3
    minimize L1+L2+L3
    subject to
        X'*a1+b1>=L1;
        L1>=X'*a2+b2;
        L1>=X'*a3+b3;
        
        Y'*a2+b2>=L2;
        L2>=Y'*a1+b1;
        L2>=Y'*a3+b3;
        
        Z'*a3+b3>=L3;
        L3>=Z'*a2+b2;
        L3>=Z'*a1+b1;
        
        L1>=1;
        L2>=1;
        L3>=1;
cvx_end;

