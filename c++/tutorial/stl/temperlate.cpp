#include<iostream>
#include<string>
#include<vector>
#include<initializer_list>

using namespace std;

template<typename T>
T square(const T& x){
    return x*x;;
}
template<typename T>
class MyVector{
public:
    MyVector()=default;
    MyVector(const initializer_list<T>& aaa){
        for(auto x:aaa){
            mdata.push_back(x);
        }
    }
    void print(){
        cout<<"----------------------"<<endl;
        for(auto x:mdata){
            cout<<x<<",";
        }
        cout<<endl;
    }

friend MyVector<T> operator*(const MyVector<T>& my,const MyVector<T>& other){
        MyVector<T>  r;
        for(unsigned i=0;i<other.mdata.size();i++){
            r.mdata.push_back(other.mdata[i]*my.mdata[i]);
        }
        return r;
    }
private:
vector<T> mdata;
};

int main(int argc, char const *argv[])
{
    cout<<square<int>(5)<<endl;
    cout<<square<double>(5.5)<<endl;

    MyVector<double> v1={1,2,3,4};
    MyVector<double> v2={2,2,1,1};
    // MyVector<double> v3=v1*v2;
    // v3.print();

    square(v1*v2).print();
    return 0;
}
