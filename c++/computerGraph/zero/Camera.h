#ifndef CAMERA_ZXK
#define CAMERA_ZXK
#include <GL/glut.h>
#include <Vector3f.h>
#include<Vector4f.h>
#include<Matrix4f.h>
#include<iostream>
using namespace std;
class Camera{
public:
    Camera();
    Camera(Vector3f,Vector3f,Vector3f);
    Matrix4f getViewMatrix() const;
    void project(float angle,float aspect,float near,float far);
private:
    Vector3f m_eyepoint;
    Vector3f m_pointTo;
    
    Vector3f m_x;
    Vector3f m_y;
    Vector3f m_z;
     
};
#endif