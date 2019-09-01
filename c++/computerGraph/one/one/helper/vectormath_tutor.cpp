#include <vecmath/Vector3f.h>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

int main(int argc, char const *argv[])
{
    Vector3f f(1,2,3);
    Vector3f c;
    cout<<"before copy"<<endl;
    cout<<f<<endl;
    cout<<c<<endl;
    cout<<f<<&f<<endl;
    cout<<c<<&c<<endl;

    cout<<"after copy"<<endl;
    c=f;
    cout<<f<<&f<<endl;
    cout<<c<<&c<<endl;

    cout<<"multi scalar,返回的是新的vector"<<endl;
    Vector3f d=c*10;
    cout<<c<<&c<<endl;
    cout<<d<<&d<<endl;

    cout<<"inline"<<endl;
    Vector3f v1(3,4,5);
    Vector3f v2=-v1;

    cout<<v1<<&v1<<endl;
    cout<<v2<<&v2<<endl;
    v1+=v2;
    cout<<v1<<&v1<<endl;
    cout<<v2<<&v2<<endl;

    cout<<"元素访问"<<endl;
    Vector3f v3(5,7,13);
    v3.print();
    cout<<v3[0]<<v3[1]<<v3[2]<<endl;
    v3.normalized();

    cout<<",单位向量,返回的立即赋值"<<endl;
    Vector3f v4(5,7,13);
    v4.print();
    v4.normalize();
    v4.print();
    cout<<"norm:"<<v4.abs()<<endl;
    v4=v4.normalized();
    v4.print();
    cout<<"norm:"<<v4.abs()<<endl;


    cout<<"dot,product"<<endl;
    Vector3f q1(0,1,2);
    Vector3f q2(3,4,5);
    cout<<"dot:"<<Vector3f::dot(q1,q2)<<endl;
    cout<<"cross:"<<Vector3f::cross(q1,q2)<<endl;
    cout<<"cross:"<<Vector3f::cross(q2,q1)<<endl;
    Vector3f normal=Vector3f::cross(q1,q2);
    cout<<"q1 cross n:"<<Vector3f::dot(q1,normal)<<endl;
    cout<<"q2 cross n:"<<Vector3f::dot(q2,normal)<<endl;
    cout<<pow(3.0,3)<<endl;

    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<float> dist(10,30);

    for(unsigned i=0;i<10;i++){
        cout<<dist(e2)<<endl;
    }
    return 0;

}
