#include<iostream>
using namespace std;
/*
 show different between delete and delete[]
 */


class FOO
{
private:
    /* data */
public:
    FOO(/* args */){};
    ~FOO(){
        cout<<"Delete Foo"<<endl;
    };
};


int main(int argc, char const *argv[])
{
    FOO* f1=new FOO();
    delete f1;
    cout<<"delete array of size 4"<<endl;
    FOO* x[4]={new FOO(),new FOO(),new FOO(),new FOO()};
    delete x[0];
    delete x[1];

    return 0;
}

 
