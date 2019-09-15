#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
using namespace std;

class Vertex{
public:
    double x;
    double y;
    double z;
    friend ostream& operator<<(ostream& out,const Vertex & v){
        out<<v.x<<","<<v.y<<","<<v.z;
        return out;
    }
};
class VertexNormal{
public:
    double x;
    double y;
    double z;
    friend ostream& operator<<(ostream& out,const VertexNormal & v){
        out<<v.x<<","<<v.y<<","<<v.z;
        return out;
    }
};

int main(int argc, char const *argv[])
{
    {
        fstream inf("garg.obj");
        int MAXBUFFER=1024;
        char buf[MAXBUFFER];
        
        cout.precision(3);
        
        while (!inf.eof())
        {
            inf.getline(buf,MAXBUFFER);
            stringstream s(buf);
            string objtype;
            s>>objtype;
            if(objtype=="v"){
                double v1,v2,v3;
                s>>v1;s>>v2;s>>v3;
                Vertex v={v1,v2,v3};
                cout<<v<<endl;
            }else if (objtype=="vn")
            {
                double v1,v2,v3;
                s>>v1;s>>v2;s>>v3;
                VertexNormal vn={v1,v2,v3};
                cout<<vn<<endl;
            }else if (objtype=="f")
            {
                /* code */
            }
            
            
        }
        
 

    }
    
    return 0;
}
