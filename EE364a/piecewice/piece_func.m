function y = piece_func(p,x)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    
    K=size(p,2);
    N=K+1;
    a=linspace(-1e-5,1,N);
    ll=a(1,1:N-1);
    uu=a(1,2:N);
    masks=bsxfun(@gt,x,ll).*bsxfun(@le,x,uu);
    X=[x ones(size(x,1),1)];
    y=sum((X*p).*masks,2);
end

