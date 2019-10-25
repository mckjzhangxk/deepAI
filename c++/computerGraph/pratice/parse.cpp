#include "parse.h"

Mesh::Mesh(const char * filename):wired(false){
    fstream fin(filename);
    if(!fin){
        cout<<filename<<" not exist!"<<endl;
        exit(0);
    }
    int MAX_SIZE=1024;;
    char buf[MAX_SIZE];
    while (fin.getline(buf,MAX_SIZE))
    {
        stringstream ss(buf);
        string type;
        ss>>type;
        if(type=="v"){
            float v1,v2,v3;
            ss>>v1;ss>>v2;ss>>v3;
            m_vertexes.push_back(Vector3f(v1,v2,v3));
            m_norms.push_back(Vector3f(0));
        }else if(type=="f"){
            int a,b,c,d,e,f,g,h,i;
            ss>>a;ss>>b;ss>>c;ss>>d;ss>>e;ss>>f;ss>>g;ss>>h;ss>>i;
            m_faces.push_back({a-1,d-1,g-1});
            m_faces_normals.push_back({c-1,f-1,i-1});
        }
    }
    
    for(Trangle tris:m_faces){
        Vector3f v1=m_vertexes[tris.a];
        Vector3f v2=m_vertexes[tris.b];
        Vector3f v3=m_vertexes[tris.c];

        Vector3f norm=Vector3f::cross(v2-v1,v3-v1);
        m_norms[tris.a]+=norm;
        m_norms[tris.b]+=norm;
        m_norms[tris.c]+=norm;
    }

    for(int i=0;i<m_norms.size();i++){
        m_norms[i]=m_norms[i].normalized();
    }

    cout<<"total vertexes #"<<m_vertexes.size()<<endl;
    cout<<"total m_norms #"<<m_norms.size()<<endl;
    cout<<"total face #"<<m_faces.size()<<endl;
}

void Mesh::setWired(bool bl){
    wired=bl;
}
void Mesh::draw(){
        glPushMatrix();
 
        // float ambient_color[4]={0.5, 0.5, 0.9, 1.0};
        // float diffuse_color[4]={0.5, 0.5, 0.9, 1.0};
        // float specular_color[4]={1.,1.,1.,1.};
        // float shineness=100;
        // glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,ambient_color);
        // glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,diffuse_color);
        // glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular_color);
        // glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,shineness);
        for(int i=0;i<m_faces.size();i++){
            Trangle faces=m_faces[i];
            Trangle norms=m_faces_normals[i];

            Vector3f v1=m_vertexes[faces.a];
            Vector3f v2=m_vertexes[faces.b];
            Vector3f v3=m_vertexes[faces.c];
            Vector3f n1=m_norms[faces.a];
            Vector3f n2=m_norms[faces.b];
            Vector3f n3=m_norms[faces.c];

            if(wired){
                drawLines({v1,v2,v3},1.f);
            }else
            {
                drawTriangle({v1,v2,v3},{n1,n2,n3});
            }
            
            
        }
        glPushMatrix();
}