#include<iostream>
using namespace std;
//https://www.youtube.com/watch?v=RC7uE_wl1Uc&list=PLE28375D4AC946CC3&index=2
class Dog{
    string name;
    int age;

public:
    Dog(){
        name="zxk";
    }
    void setAge(const int& x){
        cout<<"setAge_const"<<endl;
        age=x;
    }
    void setAge(int &x){
        cout<<"setAge_none-const"<<endl;
        age=x;
    }

    string& getName(){
        return name;
    }
    const string& getNameConst(){
        return name;
    }

       // const function
    void printDogName() const { cout << name << "_const" << endl; }
    void printDogName() { cout << getName() << " _non-const" << endl; }
};

int main(int argc, char const *argv[])
{
    Dog d;

    int age=30;
    d.setAge(age);
    d.setAge(20);

    const string& myname=d.getNameConst();
    cout<<myname<<endl;

    // string& name=d.getName();
    // name="zxxx";
    // cout<<d.getName()<<endl;

    const Dog d1;
    d1.printDogName();
    return 0;
}
