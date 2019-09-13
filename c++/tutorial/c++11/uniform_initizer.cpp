
#include<iostream>
#include<string>
#include<vector>
using namespace std;

/*
 * 2. Uniform Initialization
 */

class Dog
{
public:
    // C++ 03  Aggregate Initialization
    string name;
    int age;
    vector<double>x;
public:
    // C++ 11 extended the scope of curly brace initialization
    Dog(string name,int age){
        this->name="Dog";
        this->age=30;
    };
    Dog(const initializer_list<int> & list){}
    ~Dog(){};
friend ostream & operator<<(ostream &out,const Dog & d){
    out<<d.name<<"  "<<d.age;
    return out;
}

};

int main(int argc, char const *argv[])
{
    /* Uniform Initialization Search Order:
 * 1. Initializer_list constructor
 * 2. Regular constructor that takes the appropriate parameters.
 * 3. Aggregate initializer.
 */


    Dog d={"zxk",20};
    cout<<d<<endl;;
    return 0;
}
