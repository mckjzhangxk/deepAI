function [ dx,g,decrement ] = DiagHessionMethod(x,df,d2f )
%NEWTONMETHOD 此处显示有关此函数的摘要
%   此处显示详细说明
    g=df(x);
    Hinv=1./d2f(x);
    dx=-Hinv.*g;
    decrement=sum(-g.*dx);
end

