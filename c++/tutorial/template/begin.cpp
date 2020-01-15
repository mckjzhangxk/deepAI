#include <iostream>
using namespace std;

template<typename T,int N>
class Vector{
public:
    int getCount() const{return N;};
private: 
    T m_array[N];
};

typedef Vector<float,2> FVector2;
typedef Vector<float,3> FVector3;
typedef Vector<int,2> IVector2;
typedef Vector<int,3> IVector3;

template<class T>
void print(T x){
    x=x*1;
    cout<<x<<endl;
}
int main(int argc, char const *argv[])
{
    print(11);
    FVector2 a;
    FVector3 b;

    cout<<a.getCount()<<endl;
    cout<<b.getCount()<<endl;
    
    return 0;
}
