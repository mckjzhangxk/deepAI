function [ dx,g,decrement ] = ReusingHessianMethod(x,df,d2f,step,interval)
%NEWTONMETHOD 此处显示有关此函数的摘要
%   此处显示详细说明
    persistent L;
    if rem(step,interval)==1
        L=chol(d2f(x),'lower');
    end
    g=df(x);
    
    dx=L'\(L\-g);
    decrement=sum(-g.*dx);
end

