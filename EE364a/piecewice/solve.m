clc;clear;
%f1,f2....,f_{k+1}这些片段函数来近似f(x)
%f_j(x)=c_j*x+d_j // a_{j-1}<x<=a_{j}, c_j,d_j是要求的参数
%1.要保证convexity,c_j<=c_{j+1} //j=1....k
%2.保证连续性, c_j*a_j+d_j==c_{j+1}*a_j+d_{j+1} //j=1...k

 
for knots=1:3
    a=linspace(-1e-6,1,2+knots);
    ll=a(1,1:knots+1);
    uu=a(1,2:knots+2);

    %load data,x=(m,1),y=(m,1)
    pwl_fit_data;
    m=size(x,1);

    %mask=(m,k+1),mask(i,j)=1表示f_j用于近似(x_i,y_i)
    masks=bsxfun(@gt,x,ll).*bsxfun(@le,x,uu);


    X=[x ones(size(x,1),1)];
    Y=repmat(y,1,knots+1);

    cvx_quiet(true);
    cvx_begin
        variable p(2,knots+1);
        minimize norm((X*p-Y).*masks,2);
        subject to 
            for i=1:knots
                p(1,i)<=p(1,i+1);
                p(1,i)*a(i+1)+p(2,i)==p(1,i+1)*a(i+1)+p(2,i+1);
            end
    cvx_end
    sprintf('knots=%d,square error=%f',knots,cvx_optval^2)
    figure();
    title(['knots=' num2str(knots)])
    plot(x,y,'.',x,piece_func(p,x),'r');
    legend('groud truth','fit curve')
end