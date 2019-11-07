#ifndef PARSE
#define PARSE

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <vecmath/Vector3f.h>
#include "draw.h"
#include "material.h"
using namespace std;

struct Trangle{
    int a;
    int b;
    int c;
};
class Mesh{
public:
    Mesh(){}
    Mesh(const char * filename);
    void loadMesh(const char * filename);
    void draw(bool wired=false);

    void set_material(Material* v);
private:
    void clear();
    //m_vertexes.size==m_norms.size
    vector<Vector3f> m_vertexes;
    vector<Vector3f> m_norms;
    vector<Trangle> m_faces;

    Material* m_material;
};
#endif
