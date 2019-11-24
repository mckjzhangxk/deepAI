#ifndef PARSE
#define PARSE

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include "draw.h"
#include <vecmath.h>
#include "material.h"
#include "Object3D.h"
#include "Triangle.h"

using namespace std;

struct T{
    int a;int b;int c;
};

class Mesh:public Object3D{
public:
    Mesh(const char * filename);
    void loadMesh(const char * filename);
    void compute_norm();
    void draw(bool wired=false);
private:
    void clear();
    //m_vertexes.size==m_norms.size
    vector<int> m_vs;
    vector<T> m_fs;
    vector<Triangle> m_faces;
};
#endif
