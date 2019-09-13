#include<iostream>
#include<initializer_list>
#include<vector>
#include<algorithm>
using namespace std;
// Define your own initializer_list constructor:
class MyVector{
public:
    MyVector(const initializer_list<float>& add){
        for(auto x:add){
            cout<<x<<endl;
        }
    }
};

// Automatic normal Initialization
class Rectangle {
   public:
    // Automatic normal Initialization
    Rectangle(int height, int width, int length){ }
};


int main(int argc, char const *argv[])
{
    //C++ 03 initializer list:
    int arr[4] = {3, 2, 4, 5};
    // C++ 11 extended the support 
    vector<float> a={1,2,3,4};
    for_each(a.begin(),a.end(),[](const float & x){cout<<x<<endl;});
    
    MyVector v={1.0,1.2};
    
    Rectangle r({1,2,3});
    return 0;
}
