#include<vector>
#include<vecmath/Matrix4f.h>
#include<vecmath/Vector3f.h>
#include<GL/glut.h>
#include<iostream>
using namespace std;


void drawTriangle(const vector<vector<Vector3f> > pts,const Vector3f &color);
void drawLines(const vector<Vector3f> pts,const Vector3f &color,GLfloat linewidth=2);