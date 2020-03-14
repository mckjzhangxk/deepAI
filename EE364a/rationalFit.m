clc;clear;
k=201;
i=(1:1:k)';
t=-3+6*(i-1)/(k-1);
T=[ones(k,1) t t.^2];
y=exp(t);

u=exp(3);l=0;

while u-l>1e-3
    gamma=(l+u)/2;

    cvx_begin
        cvx_quiet(true);
        variable a(3);
        variable b(2);
        subject to
        abs(T*a-y.*(T*[1;b]))<=gamma*(T*[1;b]);
    cvx_end
    if strcmp('Solved',cvx_status)
        u=gamma;
        a_opt=a;
        b_opt=b;
        objval_opt=gamma;
    else
        l=gamma;
    end
end

figure(1);
y_fit=(T*a_opt)./(T*[1;b_opt]);
plot(t,y,'b',t,y_fit,'r+');
xlabel('t');
figure(2);
plot(t,abs(y-y_fit));
xlabel('t');
ylabel('error')