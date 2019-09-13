#include<iostream>
using namespace std;

class LinearImpl
{
private:
    /* data */
public:
    LinearImpl(int M,int N){};
    ~LinearImpl(){};
};



class Linear
{
private:
    LinearImpl mlinear;
public:
    Linear():mlinear(LinearImpl(2,3)){};
    ~Linear(){};

};

int main(int argc, char const *argv[])
{
    
    return 0;
}
