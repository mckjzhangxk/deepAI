function y = clamp( x ,t)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
    y=min(x,t);
    y=max(y,-t);
end

