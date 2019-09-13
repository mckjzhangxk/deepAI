#include<iostream>
#include<string>
using namespace std;

/*
 * 10. final (for virtual function and for class)
 */
// no class can be derived from Dog
class Dog final{

};
// class YellowDog:public Dog{}

class Cat{
public:
    Cat()=default;
    Cat(const char * name){
        this->name=name;
    };
protected:
    virtual void eat() final{
        cout<<name<<" eat"<<endl;
    };
private:
    string name="zxk";
};
class SmallCat:public Cat{
public:
    SmallCat(const char * name):Cat(name){
    };

    void happy(){
        
        this->eat();
    }
};

int main(int argc, char const *argv[])
{
    SmallCat c("sss");
    c.happy();
    return 0;
}
