#include<Matrix4f.h>
#include<Vector4f.h>
#include<iostream>
using namespace std;

int main(int argc, char const *argv[])
{
    Matrix4f f=Matrix4f::identity();
    Vector4f v(1,2,3,4);
    f.setCol(3,v);
    f.print();
    return 0;
}
