function [ dx,g,decrement ] = NewtonMethod(x,df,d2f )
%NEWTONMETHOD 此处显示有关此函数的摘要
%   此处显示详细说明
    g=df(x);
    H=d2f(x);
    dx=-H\g;
    decrement=sum(-g.*dx);
end

