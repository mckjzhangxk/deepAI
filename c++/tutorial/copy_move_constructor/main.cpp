#include<iostream>
using namespace std;


class MyData
{
private:
    /* data */
public:
    MyData(){};
    //copy constructor,激活这个函数的方法
    //MyData a=b;
    //或者 foo(a),foo的函数原型是foo(MyData)
    MyData(MyData &);

    MyData operator=(MyData & x){
        cout<<"============="<<endl;
    }
    ~MyData(){};
};

 
MyData::MyData(MyData &d){
    cout<<"call lvalue constructor"<<endl;
}


void foo(MyData d){}
void bar(MyData& d){}

// void foo(MyData &&d){}

int main(int argc, char const *argv[])
{    
    MyData x;
    foo(x);//调用参数，复制参数的时候，lvalue constructor

    cout<<"construct with assign "<<endl;
    MyData y=x;//同样，这也是调用构造器,而不是赋值

    cout<<"without construct,by reference"<<endl;
    bar(x);

    MyData z;
    z=x;
    return 0;
}
