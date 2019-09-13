#include<iostream>
#include<string>
using namespace std;


class Animal
{
private:
    
public:
    Animal(){};
    ~Animal(){};
    virtual void run(int x){
        cout<<"animal run"<<endl;
    };
   virtual void eat( char* x) const{
       cout<<"animal eat"<<endl;
   };
};

class Dog:public Animal{
public:
    // C++ 11
     // Error: no function to override
     void run(float x)  override{
            cout<<"dog run"<<endl;
     };
     // Error: no function to override
     void eat( char * x) override{
         cout<<"dog eat"<<endl;
     } ;
};
int main(int argc, char const *argv[])
{
    Dog d;
    d.eat("x");
    d.run(1);
    return 0;
}

