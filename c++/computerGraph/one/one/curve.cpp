#include "curve.h"
#include "extra.h"
#ifdef WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include<cmath>
#include<random>
using namespace std;

namespace
{
    // Approximately equal to.  We don't want to use == because of
    // precision issues with floating point.
    inline bool approx( const Vector3f& lhs, const Vector3f& rhs )
    {
        const float eps = 1e-8f;
        return ( lhs - rhs ).absSquared() < eps;
    }

     Vector3f initBiNormal(const vector<Vector3f> &P){

        if(P[0]==Vector3f(0,0,1)){
            return Vector3f(0,0,1);
        }

        random_device rd;
        mt19937 e2(rd());
        uniform_real_distribution<float> dist(0,1);
        float x,y,z;

        x=dist(e2);
        y=dist(e2);
        z=dist(e2);
        
        Vector3f r(x,y,z);
        r.normalize();
        return r;
    } 
    vector<Vector3f> changeControlPoints(const Vector3f& p0,
                                        const Vector3f& p1,
                                        const Vector3f& p2,
                                        const Vector3f& p3){
        vector<Vector3f> ret;
        //准备B_spline和B_bezier_inv
        Matrix4f B_bezier(1,-3,3,-1,0,3,-6,3,0,0,3,-3,0,0,0,1);
        bool sigular;
        Matrix4f B_bezier_inv=B_bezier.inverse(&sigular,1e-5);
        Matrix4f B_spline=Matrix4f(1,-3,3,-1,4,0,-6,3,1,3,3,-3,0,0,0,1);
        B_spline/=6.0f;

        Matrix4f M=B_spline*B_bezier_inv;
        
        for(unsigned i=0;i<4;i++){
            ret.push_back(p0*M(0,i)+p1*M(1,i)+p2*M(2,i)+p3*M(3,i));
        }
        return ret;
    }  


}
Curve evalBezier( const vector< Vector3f >& P, unsigned steps )
{
    // Check
    if( P.size() < 4 || P.size() % 3 != 1 )
    {
        cerr << "evalBezier must be called with 3n+1 control points." << endl;
        exit( 0 );
    }

    // TODO:
    // You should implement this function so that it returns a Curve
    // (e.g., a vector< CurvePoint >).  The variable "steps" tells you
    // the number of points to generate on each piece of the spline.
    // At least, that's how the sample solution is implemented and how
    // the SWP files are written.  But you are free to interpret this
    // variable however you want, so long as you can control the
    // "resolution" of the discretized spline curve with it.

    // Make sure that this function computes all the appropriate
    // Vector3fs for each CurvePoint: V,T,N,B.
    // [NBT] should be unit and orthogonal.

    // Also note that you may assume that all Bezier curves that you
    // receive have G1 continuity.  Otherwise, the TNB will not be
    // be defined at points where this does not hold.

    cerr << "\t>>> evalBezier has been called with the following input:" << endl;

    cerr << "\t>>> Control points (type vector< Vector3f >): "<< endl;
    for( unsigned i = 0; i < P.size(); ++i )
    {
        // cerr << "\t>>> " << P[i] << endl;
        P[i].print();
    }

    cerr << "\t>>> Steps (type steps): " << steps << endl;
    // cerr << "\t>>> Returning empty curve." << endl;

    // Right now this will just return this empty curve.
    Curve R(steps+1);

    Vector3f B=initBiNormal(P);
    int offset=0;

    while (offset<P.size()-1)
    {
        for(unsigned s=0;s<steps+1;s++){
            float t=s/(float(steps));
            float weight_V[4]={
                pow(1-t,3),
                3*t*pow(1-t,2),
                3*(1-t)*pow(t,2),
                pow(t,3)
            };
            float weight_T[4]={
                -3*pow(1-t,2),
                3-12*t+9*pow(t,2),
                6*t-9*pow(t,2),
                3*pow(t,2)
            };
            struct CurvePoint p;
            p.V=Vector3f();
            p.T=Vector3f();

            for(unsigned i=0;i<4;i++){
                p.V+=P[offset+i]*weight_V[i];
                p.T+=P[offset+i]*weight_T[i];
            }
            // 切线均一化
            p.T.normalize();

            //法线是之前的BxT,然后均已化,能不能TxB呢?
            p.N=Vector3f::cross(B,p.T);
            p.N.normalize();
            
            //新的B 是TxN 
            B=Vector3f::cross(p.T,p.N);
            B.normalize();
            p.B=B;
            
            R[s]=p;
        }
        offset+=3;
    }
    
    

    return R;
}

Curve evalBspline( const vector< Vector3f >& P, unsigned steps )
{
    // Check
    if( P.size() < 4 )
    {
        cerr << "evalBspline must be called with 4 or more control points." << endl;
        exit( 0 );
    }

    // TODO:
    // It is suggested that you implement this function by changing
    // basis from B-spline to Bezier.  That way, you can just call
    // your evalBezier function.

    cerr << "\t>>> evalBSpline has been called with the following input:" << endl;

    cerr << "\t>>> Control points (type vector< Vector3f >): "<< endl;
    for( unsigned i = 0; i < P.size(); ++i )
    {
        cerr << "\t>>> " << P[i] << endl;
    }

    cerr << "\t>>> Steps (type steps): " << steps << endl;
    cerr << "\t>>> Returning empty curve." << endl;

    // Return an empty curve right now.
    vector<Vector3f> Pnew;

    for(unsigned i=0;i<P.size()-4+1;i++){
        vector<Vector3f> newpts=changeControlPoints(P[i],P[i+1],P[i+2],P[i+3]);
        
        for(unsigned j=0;j<4;j++)
            Pnew.push_back(newpts[j]);
       
    }
  
    Curve r=evalBezier(Pnew,steps);
    return r;
}

Curve evalCircle( float radius, unsigned steps )
{
    // This is a sample function on how to properly initialize a Curve
    // (which is a vector< CurvePoint >).
    
    // Preallocate a curve with steps+1 CurvePoints
    Curve R( steps+1 );

    // Fill it in counterclockwise
    for( unsigned i = 0; i <= steps; ++i )
    {
        // step from 0 to 2pi
        float t = 2.0f * M_PI * float( i ) / steps;

        // Initialize position
        // We're pivoting counterclockwise around the y-axis
        R[i].V = radius * Vector3f( cos(t), sin(t), 0 );
        
        // Tangent vector is first derivative
        R[i].T = Vector3f( -sin(t), cos(t), 0 );
        
        // Normal vector is second derivative
        R[i].N = Vector3f( -cos(t), -sin(t), 0 );

        // Finally, binormal is facing up.
        R[i].B = Vector3f( 0, 0, 1 );
    }

    return R;
}

void drawCurve( const Curve& curve, float framesize )
{
    // Save current state of OpenGL
    glPushAttrib( GL_ALL_ATTRIB_BITS );

    // Setup for line drawing
    glDisable( GL_LIGHTING ); 
    glColor4f( 1, 1, 1, 1 );
    glLineWidth( 1 );
    
    // Draw curve
    glBegin( GL_LINE_STRIP );
    for( unsigned i = 0; i < curve.size(); ++i )
    {
        glVertex( curve[ i ].V );
    }
    glEnd();

    glLineWidth( 1 );

    // Draw coordinate frames if framesize nonzero
    if( framesize != 0.0f )
    {
        Matrix4f M;

        for( unsigned i = 0; i < curve.size(); ++i )
        {
            M.setCol( 0, Vector4f( curve[i].N, 0 ) );
            M.setCol( 1, Vector4f( curve[i].B, 0 ) );
            M.setCol( 2, Vector4f( curve[i].T, 0 ) );
            M.setCol( 3, Vector4f( curve[i].V, 1 ) );

            glPushMatrix();
            glMultMatrixf( M );
            glScaled( framesize, framesize, framesize );
            glBegin( GL_LINES );
            glColor3f( 1, 0, 0 ); glVertex3d( 0, 0, 0 ); glVertex3d( 1, 0, 0 );
            glColor3f( 0, 1, 0 ); glVertex3d( 0, 0, 0 ); glVertex3d( 0, 1, 0 );
            glColor3f( 1, 0, 1 ); glVertex3d( 0, 0, 0 ); glVertex3d( 0, 0, 1 );
            //一下是练习,GL_LINES是一条一条的划线,一次指定2个点
            // glColor3f(1,0,0);glVertex3d(0,0,10);glVertex3d(0,10,0);
            // glColor3f(0,1,0);glVertex3d(0,10,0);glVertex3d(10,0,0);
            // glColor3f(1,0,1);glVertex3d(10,0,0);glVertex3d(0,0,10);
            glEnd();
            glPopMatrix();
        }
    }
    
    // Pop state
    glPopAttrib();
}

