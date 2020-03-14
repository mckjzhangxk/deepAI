function dy = fobj_grad(  A,x)
%UNTITLED3 此处显示有关此函数的摘要
% assume x in domain
% Ax<1 && x.^2<1

    y1=1./(1-A*x);
    dy1=sum(bsxfun(@times,A,y1),1)';
    dy2=2*x./(1-x.^2);
    dy=dy1+dy2;    
end

