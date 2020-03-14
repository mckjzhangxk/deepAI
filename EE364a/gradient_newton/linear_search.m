function t = linear_search(f,x,dx,df,alpha,beta)
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
    t=1;
    fx=f(x);
    k=alpha*sum(df.*dx);
    
    %define a good linear upper bound
    U=@(t)fx+t*k;
    
    while f(x+t*dx)>U(t)
        t=t*beta;
    end

end

