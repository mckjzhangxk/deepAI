#include<iostream>
#include<string>
using namespace std;


class Dog
{
private:
    /* data */
public:
    Dog(/* args */){
        cout<<"Dog1"<<endl;
    };
    // C++ 11:先执行Dog,然后函数体
    Dog(int a):Dog(){

        cout<<"Dog2"<<endl;
    };
    ~Dog(){};
};

int main(int argc, char const *argv[])
{
    Dog(2);

    Dog();
    return 0;
}

