#include "Camera.h"

Camera::Camera():Camera(Vector3f(0,0,5),Vector3f(0,0,0),Vector3f(0,1,0)){
}
Camera::Camera(Vector3f eye,Vector3f pointTo,Vector3f up){
    m_eyepoint=eye;
    m_pointTo=pointTo;

    m_z=(pointTo-eye).normalized();
    m_x=Vector3f::cross(m_z,up).normalized();
    m_y=Vector3f::cross(m_x,m_z).normalized();
}
Matrix4f Camera::getViewMatrix() const{
    Matrix4f ret=Matrix4f::identity();


    ret.setCol(0,Vector4f(m_x,0));
    ret.setCol(1,Vector4f(m_y,0));
    ret.setCol(2,Vector4f(m_z,0));
    ret.setCol(3,Vector4f(m_eyepoint,1));
    
    return ret.inverse();
}

void Camera::project(float angle,float aspect,float near,float far){
    cout<<aspect<<endl;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // 50 degree fov, uniform aspect ratio, near = 1, far = 100
    gluPerspective(60, 1., 1.0, 100.0);
}