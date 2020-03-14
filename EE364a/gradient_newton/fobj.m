function y = fobj( A,x )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    
    Ax=A*x;
    X2=x.^2;
    %every x ood will return inf
    if (max(Ax)>=1) || max(X2)>=1
        y=inf;
        return
    end
    
    y1=-sum(log(1-Ax));
    y2=-sum(log(1-x.^2));
    y=y1+y2;
end

