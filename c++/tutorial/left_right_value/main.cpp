#include <iostream>
using namespace std;
/*
https://www.youtube.com/watch?v=UTUdhjzws5g

lvalue和rvalue
1.lvalue表示所有可以被程序引用的 对象
2.和lvalue相对的是rvalue

*/
//返回的是rvalue
int sum(int x,int y){
    return x+y;
}
//x是lvalue reference,所以不可以foo(22),因为22是rvalue
void foo(int& x){}
//同上，但是const reference可以引用一个rvalue
int square(const int &x){
    return x*x;
}

//返回一个lvalue reference,所有函数不一定都是返回rvalue,但是注意，如果
// 为返回的是局部变量的reference，再去调用有可能出错
// int &x=bar();x=22; error

int global=1000;
int& bar(){
    return global;
}
int& bar1(){
    int a;
    return a;
}
class Dog{
public:
    Dog(string name){
        this->name=name;
    }
    string name;
    void setName(string name){
        this->name=name;
    }
};
int main()
{
    //lvalue example
    int a=0;
    int b[3]={1,2,3};  
    
    //左边的是lvalue,右边的是rvalue
    Dog d=Dog("zxk");
    
    //rvalue example
    // int *p=&(a+0); //a+0没发引用
    int *p=&a;


    //lvalue reference
    int &c=a;
    
    // 以下都不能编译通过，因为把rvalue赋值给lvalue reference
    // int &d=sum(1,2);
    // int &d=22;
    // foo(sum(1,2));
    //只有const lvalue reference是个例外，我可以复制一个rvalue给const lvalue reference
    // 但其实是把rvalue先转成lvalue
    const int &exception=sum(1,2);
    square(2);

    //误区1,函数返回的都是rvalue
    foo(bar());
    bar()=22;
    //编译通过，但是实际保存，因为返回的是local variable的lvalue
    int abc=bar1();
    //误区2,rvalue不能修改
    Dog("x").setName("xx");
    
    
    return 0;
}