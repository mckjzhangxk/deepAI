#ifndef PARSE
#define PARSE

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <vecmath/Vector3f.h>
#include "draw.h"

using namespace std;

struct Trangle{
    int a;
    int b;
    int c;
};
class Mesh{
public:
    Mesh(const char * filename);
    void draw();
private:
    vector<Vector3f> m_vertexes;
    vector<Vector3f> m_norms;
    vector<Trangle> m_faces;
    vector<Trangle> m_faces_normals;
};
#endif