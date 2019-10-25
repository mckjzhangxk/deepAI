#include "Camera.h"

Camera::Camera():Camera(Vector3f(0,0,5),Vector3f(0,0,0),Vector3f(0,1,0)){
}
Camera::Camera(Vector3f eye,Vector3f pointTo,Vector3f up){
    m_eyepoint=eye;
    m_pointTo=pointTo;

    m_z=(eye-pointTo).normalized();
    m_x=Vector3f::cross(up,m_z).normalized();
    m_y=Vector3f::cross(m_z,m_x).normalized();
}

void Camera::setEyePoint(Vector3f & eye){
    m_eyepoint=eye;
}
Vector3f Camera::getEyePoint() const{
    return m_eyepoint;
}
void Camera::setCenter(Vector3f & center){
     m_pointTo=center;
}
Vector3f Camera::getCenter() const{
    return m_pointTo;
}

Matrix4f Camera::getViewMatrix() const{
    Matrix4f ret=Matrix4f::identity();


    ret.setCol(0,Vector4f(m_x,0));
    ret.setCol(1,Vector4f(m_y,0));
    ret.setCol(2,Vector4f(m_z,0));
    ret.setCol(3,Vector4f(m_eyepoint,1));
    
    return ret.inverse();
}

void Camera::perspective_projection(float angle,float aspect,float near,float far){
   
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    Matrix4f m=Matrix4f::perspectiveProjection(M_PI*angle/180.,aspect,near,far,false);
    glLoadMatrixf(m);
}