#include "surf.h"
#include "extra.h"
#include <cmath>
using namespace std;

namespace
{
    
    // We're only implenting swept surfaces where the profile curve is
    // flat on the xy-plane.  This is a check function.
    static bool checkFlat(const Curve &profile)
    {
        for (unsigned i=0; i<profile.size(); i++)
            if (profile[i].V[2] != 0.0 ||
                profile[i].T[2] != 0.0 ||
                profile[i].N[2] != 0.0)
                return false;
    
        return true;
    }

    void calcFaces(unsigned Usize,unsigned Vsize,Surface& surf){
    
        for(unsigned u=0;u<Usize;u++)
            for(unsigned v=0;v<Vsize-1;v++){
                /*
                a    c
                
                b    d
                 */
                
                int a=u*Vsize+v;
                int b=a+1;
                int c=((u+1)%Usize)*Vsize+v;
                int d=c+1;

                surf.VF.push_back(Tup3u(a,b,d));
                surf.VF.push_back(Tup3u(a,d,c));
            }
    }
}

Surface makeSurfRev(const Curve &profile, unsigned steps)
{
    Surface surface;
    
    if (!checkFlat(profile))
    {
        cerr << "surfRev profile curve must be flat on xy plane." << endl;
        exit(0);
    }

    // TODO: Here you should build the surface.  See surf.h for details.

    cerr << "\t>>> makeSurfRev called (but not implemented).\n\t>>> Returning empty surface." << endl;
    
    float pi=3.1415926;
    float theta=2*pi/((float)steps);
    int curveSize=profile.size();

    for(unsigned i=0;i<steps;i++){
        for(CurvePoint p:profile){
            float c=cos(theta*i);
            float s=sin(theta*i);
            Matrix3f M(c,0,s,0,1,0,-s,0,c);
            
            Vector3f V=M*p.V;
            // 这里对一个ratation matrix inverse and transpose,相当于没有变化
            bool sigular;        
            Matrix3f Minv_T=M.inverse(&sigular,1e-4);
            Minv_T.transpose();

            Vector3f N=Minv_T*p.N;
            
            surface.VV.push_back(V);
            surface.VN.push_back(-N);
            // (i,point_index)
        }
    
    }
    
    cout<<"应该有"<<(steps*curveSize)<<"个点，实际有"<<surface.VV.size()<<"个点"<<endl;;
                //我在这里犯了个错误，误认为图是从上往下建立的，而实际是自下而上，四个点的关系是
            /*
            b    d

            a    c
            但是之后我反转了反向，相当于从上往下够
            a    c

            b    d
            */
           
//    for(unsigned u=0;u<steps;u++){
//         for(unsigned v=0;v<curveSize;v++){
//             int a=u*curveSize+v;
//             int b=a+1;
//             int c=((u+1)%steps)*curveSize+v;
//             int d=c+1;


//             surface.VF.push_back(Tup3u(a,d,c));
//             surface.VF.push_back(Tup3u(a,b,d));
//         }
//     }


    
    calcFaces(steps,curveSize,surface);
    cout<<"面："<<surface.VF.size()<<endl;
    return surface;
}

Surface makeGenCyl(const Curve &profile, const Curve &sweep )
{
    Surface surface;

    if (!checkFlat(profile))
    {
        cerr << "genCyl profile curve must be flat on xy plane." << endl;
        exit(0);
    }

    // TODO: Here you should build the surface.  See surf.h for details.

    cerr << "\t>>> makeGenCyl called (but not implemented).\n\t>>> Returning empty surface." <<endl;
    bool sigular;
    for(CurvePoint cp:sweep){
        Matrix3f M(cp.N,cp.B,cp.T);
        Matrix3f MintT=M.inverse(&sigular,1e-4).transposed();
       
        for(CurvePoint p:profile){
            Vector3f vp=M*p.V+cp.V;
            Vector3f vn=MintT*p.N;
            surface.VV.push_back(vp);
            surface.VN.push_back(-vn);
        }
    }
    cout<<"应该有"<<(sweep.size()*profile.size())<<"个点，实际有"<<surface.VV.size()<<"个点"<<endl;

    calcFaces(sweep.size(),profile.size(),surface);
    return surface;
}

void drawSurface(const Surface &surface, bool shaded)
{
    // Save current state of OpenGL
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    if (shaded)
    {
        // This will use the current material color and light
        // positions.  Just set these in drawScene();
        glEnable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // This tells openGL to *not* draw backwards-facing triangles.
        // This is more efficient, and in addition it will help you
        // make sure that your triangles are drawn in the right order.
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
    }
    else
    {        
        glDisable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        
        glColor4f(0.4f,0.4f,0.4f,1.f);
        glLineWidth(1);
    }

    glBegin(GL_TRIANGLES);
    for (unsigned i=0; i<surface.VF.size(); i++)
    {
        glNormal(surface.VN[surface.VF[i][0]]);
        glVertex(surface.VV[surface.VF[i][0]]);
        glNormal(surface.VN[surface.VF[i][1]]);
        glVertex(surface.VV[surface.VF[i][1]]);
        glNormal(surface.VN[surface.VF[i][2]]);
        glVertex(surface.VV[surface.VF[i][2]]);
    }
    glEnd();

    glPopAttrib();
}

void drawNormals(const Surface &surface, float len)
{
    // Save current state of OpenGL
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_LIGHTING);
    glColor4f(0,1,1,1);
    glLineWidth(1);

    glBegin(GL_LINES);
    for (unsigned i=0; i<surface.VV.size(); i++)
    {
        glVertex(surface.VV[i]);
        glVertex(surface.VV[i] + surface.VN[i] * len);
    }
    glEnd();

    glPopAttrib();
}

void outputObjFile(ostream &out, const Surface &surface)
{
    
    for (unsigned i=0; i<surface.VV.size(); i++)
        out << "v  "
            << surface.VV[i][0] << " "
            << surface.VV[i][1] << " "
            << surface.VV[i][2] << endl;

    for (unsigned i=0; i<surface.VN.size(); i++)
        out << "vn "
            << surface.VN[i][0] << " "
            << surface.VN[i][1] << " "
            << surface.VN[i][2] << endl;

    out << "vt  0 0 0" << endl;
    
    for (unsigned i=0; i<surface.VF.size(); i++)
    {
        out << "f  ";
        for (unsigned j=0; j<3; j++)
        {
            unsigned a = surface.VF[i][j]+1;
            out << a << "/" << "1" << "/" << a << " ";
        }
        out << endl;
    }
}
