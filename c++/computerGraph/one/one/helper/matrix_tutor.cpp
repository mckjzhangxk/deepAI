#include <vecmath/Matrix3f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Matrix4f.h>

#include <iostream>
using namespace std;
int main(int argc, char const *argv[])
{
    cout<<"Base matrix"<<endl;
    Matrix3f m1(1,2,3,4,5,6,7,8,9);
    Vector3f v1(1,2,3);

    cout<<m1*v1<<endl;
    (m1*m1).print();

    cout<<"elements"<<endl;
    Matrix3f m2(1,2,3,4,5,6,7,8,9);
    m2.print();
    cout<<m2(1,2)<<endl;
    cout<<m2.getRow(1)<<endl;
    cout<<m2.getCol(0)<<endl;

    cout<<"create matrix from vector"<<endl;
    Vector3f p1(0,1,2);
    Vector3f p2(3,4,5);
    Vector3f p3(6,7,8);
    Matrix3f mrow(p1,p2,p3,false);
    Matrix3f mcol(p1,p2,p3,true);
    mrow.print();
    mcol.print();

    cout<<endl;
    cout<<"determine,inverse"<<endl;
    // Matrix3f ma(2,-1,0,-1,2,-1,0,-1,2);
    Matrix3f ma(1,2,3,4,5,6,7,8,9);
    
    bool sigular;
    cout<<"det:"<<ma.determinant()<<endl;
    Matrix3f inv=ma.inverse(&sigular,1e-5);
    cout<<"sigular:"<<sigular<<endl;
    inv.print();

    cout<<endl;
    cout<<"to important matrix"<<endl;

    Matrix4f B_bezier(1,-3,3,-1,0,3,-6,3,0,0,3,-3,0,0,0,1);
    Matrix4f B_bezier_inv=B_bezier.inverse(&sigular,1e-5);
    Matrix4f B_spline=Matrix4f(1,-3,3,-1,4,0,-6,3,1,3,3,-3,0,0,0,1);
    B_spline/=6;

    Vector3f p(1,2,3);
   
   
    B_spline/=(1/6.0f);
    cout<<"B_bezier"<<endl;
    B_bezier.print();
    cout<<endl;

    cout<<"B_spline"<<endl;
    B_spline.print();
    cout<<endl;

    cout<<"B_bezier_inv"<<endl;
    B_bezier_inv.print();
    cout<<endl;
    
    Matrix4f M=B_spline*B_bezier_inv;

    cout<<"convert matrix"<<endl;
    M.print();
    cout<<endl;

    return 0;
}
