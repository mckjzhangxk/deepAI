#include "mesh.h"

Mesh::Mesh(const char * filename){
    loadMesh(filename);
}




void Mesh::clear()
{
    m_faces.clear();
    m_vs.clear();
    m_fs.clear();
}

void Mesh::loadMesh(const char *filename)
{
    clear();
    std::ifstream f ;
        f.open(filename);
        if(!f.is_open()) {
            std::cout<<"Cannot open "<<filename<<"\n";
            return;
        }
        std::string line;
        std::string vTok("v");
        std::string fTok("f");
        std::string texTok("vt");
        char bslash='/',space=' ';
        std::string tok;
        vector<Vector3f> v;
        vector<Vector2f> texCoord;

        int idx=0;
        while(1) {
            std::getline(f,line);
            if(f.eof()) {
                break;
            }
            if(line.size()<3) {
                continue;
            }
            if(line.at(0)=='#') {
                continue;
            }
            std::stringstream ss(line);
            ss>>tok;
            if(tok==vTok) {
                Vector3f vec;
                ss>>vec[0]>>vec[1]>>vec[2];
                v.push_back(vec);

                m_vs.push_back(idx);
                idx++;
            } else if(tok==fTok) {
                if(line.find(bslash)!=std::string::npos) {
                    std::replace(line.begin(),line.end(),bslash,space);
                    std::stringstream facess(line);
                    facess>>tok;

                    int a,b,c,d,e,f;
                    facess>>a;facess>>b;facess>>c;facess>>d;facess>>e;facess>>f;
                    Triangle trig(v[a-1],v[c-1],v[e-1]);
                    Vector2f ts[3]={texCoord[b-1],texCoord[d-1],texCoord[f-1]};
                    trig.setTextCoords(ts);
                    m_faces.push_back(trig);

                    m_fs.push_back({a-1,c-1,e-1});
                } else {
                    int a,b,c;
                    ss>>a;ss>>b;ss>>c;
                    Triangle trig(v[a-1],v[b-1],v[c-1]);

                    m_faces.push_back(trig);
                    m_fs.push_back({a-1,b-1,c-1});
                }
            } else if(tok==texTok) {
                Vector2f texcoord;
                ss>>texcoord[0];
                ss>>texcoord[1];
                texCoord.push_back(texcoord);
            }
        }
        f.close();
        compute_norm();
}

void Mesh::compute_norm()
{
    vector<Vector3f> n;
    n.resize(m_vs.size());
    for(unsigned int ii=0; ii<m_fs.size(); ii++) {
        Vector3f nm=m_faces[ii].compute_norm();
        int a=m_fs[ii].a;int b=m_fs[ii].b;int c=m_fs[ii].c;
        n[a]+=nm;n[b]+=nm;n[c]+=nm;
    }

    for(unsigned int ii=0; ii<n.size(); ii++) {
        n[ii].normalize();
    }
    for(unsigned int ii=0; ii<m_fs.size(); ii++) {
        int a=m_fs[ii].a;int b=m_fs[ii].b;int c=m_fs[ii].c;
        Vector3f nms[3]={n[a],n[b],n[c]};
        m_faces[ii].setNormals(nms);
    }
}


void Mesh::draw(bool wired){

        if(m_material)
            m_material->loadMaterial();
        for(auto i=0;i<m_faces.size();i++){
            Triangle faces=m_faces[i];
            faces.draw(wired);
        }

}
