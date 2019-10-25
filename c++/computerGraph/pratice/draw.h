#include<vector>
#include<vecmath/Matrix4f.h>
#include<vecmath/Vector3f.h>
#include<GL/glut.h>
#include<iostream>
#include <cassert>

using namespace std;

/*
pts must have 3 elements,so does norms
*/
void drawTriangle(const vector<Vector3f > pts,const vector<Vector3f > norms);
void drawLines(const vector<Vector3f> pts,GLfloat linewidth);