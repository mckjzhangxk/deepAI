#include<iostream>
#include<memory>

using namespace std;

class Report
{
private:

public:
    Report()=default;
    ~Report(){
        cout<<"delete object"<<endl;
    };
};

unique_ptr<Report> foo(){
    unique_ptr<Report> a(new Report);
    return a;
}
int main(int argc, char const *argv[])
{
    {
        cout<<"native"<<endl;
        Report *p=new Report();
        delete p;
    }
    
    {
        cout<<"unique"<<endl;
        //对unique赋值或者 调用构造函数,必须使用rvalue
        unique_ptr<Report> p(new Report());
        unique_ptr<Report> p1=move(p);
        p1=foo();
    }
    {
        cout<<"share"<<endl;
        //p,p1都是指向同一个对象,计数器是2,计算器到0的时候,才会执行delet
        shared_ptr<Report> p(new Report());
        shared_ptr<Report> p1=p;
   
    }
    return 0;
}
