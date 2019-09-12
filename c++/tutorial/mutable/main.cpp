#include<iostream>
#include<vector>
using namespace std;

class My
{
private:
    //mutable可以让const函数修改counter
    mutable int counter;
    int *v1;
    vector<int> v2;    
public:
    My();
    ~My();
    int getValue(int index) const{
        counter++;
        // const_cast<My*>(this)->counter+=1;
        return v2[index];
    }
    int getValue1(int index) const{
        //const表示成员是const,而不数数据是const
        v1[index]+=1;
        return v1[index];
    }
};


int main(int argc, char const *argv[])
{
    
    return 0;
}

 