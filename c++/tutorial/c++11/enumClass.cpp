
#include<iostream>
#include<string>
using namespace std;

// C++ 03
// enum  Address{HOME,SCHOOL};
// enum  Fruit{APPPLE,ORANGE};
//c++ 11
enum class Address{HOME,SCHOOL};
enum class Fruit{APPPLE,ORANGE};

int main(int argc, char const *argv[])
{
    Fruit f=Fruit::APPPLE;
    Address ad=Address::HOME;

    // Compile fails because we haven't define ==(Address, Fruit)
    cout<<(f==ad)<<endl;
    return 0;
}
