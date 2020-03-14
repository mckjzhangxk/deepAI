function [opt_x history ts ] = solver(f,x,method,ALPHA,BETA,eps,MAXIters)
%SOLVER 此处显示有关此函数的摘要
%   此处显示详细说明
history=[];
ts=[];
for i=1:MAXIters
    %find good direction 
    
    [dx,g,decrement]=method(x,i);
    %find best step size
    t=linear_search(f,x,dx,g,ALPHA,BETA);
    %update x
    x=x+t*dx;
    
    if decrement<eps
        break;
    end
    
    history=[history f(x)];
    ts=[ts t];
end

opt_x=x;
end

