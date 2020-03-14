function dy2 = fobj_diag_hessian( A,x )
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明

    y1=1./(1-A*x);
    L=bsxfun(@times,A,y1);
    
    dy2_1=sum(L.^2,1)';
    
    
    dy2_2=2*(1+x.^2)./((1-x.^2).^2);
    
    dy2=dy2_1+dy2_2;
end

