#include<iostream>
#include<string>
using namespace std;
int main(int argc, char const *argv[])
{
    string s1("hello world");

    cout<<"size:"<<s1.size()<<endl;
    cout<<"capacity:"<<s1.capacity()<<endl;

    s1.reserve(100);
    cout<<"after reserve 100,capacity:"<<s1.capacity()<<endl;
    cout<<"size:"<<s1.size()<<endl;

    s1.shrink_to_fit();
    cout<<"after shrink,capacity:"<<s1.capacity()<<endl;
    cout<<"size:"<<s1.size()<<endl;

    string s2{'a','b','c'};
    return 0;
}
