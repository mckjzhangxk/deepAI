#include "Camera.h"

void createLocalFrame(  const Vector3f& eyepoint,
                        const Vector3f& center,
                        const Vector3f& up_direction,
                        Vector3f & x,Vector3f & y,Vector3f & z){
    z=(eyepoint-center).normalized();
    x=Vector3f::cross(up_direction,z).normalized();
    y=Vector3f::cross(z,x).normalized();

}
Camera::Camera():Camera(Vector3f(0,0,5),Vector3f(0,0,0),Vector3f(0,1,0)){

}
Camera::Camera(Vector3f eye,Vector3f pointTo,Vector3f up):m_oper_type(CONST_NONE){
    m_eyepoint=eye;
    m_center=pointTo;
    m_up=up;

    compute_viewMatrix();
}
void Camera::setDimension(int W,int H){
    m_screenW=W;
    m_screenH=H;
}
void Camera::setEyePoint(Vector3f & eye){
    m_eyepoint=eye;
    compute_viewMatrix();
}
void Camera::setEyePoint(Vector3f && eye){
    setEyePoint(eye);
}

Vector3f Camera::getEyePoint() const{
    return m_eyepoint;
}
void Camera::setCenter(Vector3f & center){
     m_center=center;
     compute_viewMatrix();
}
void Camera::setCenter(Vector3f && center){
     setCenter(center);
}

Vector3f Camera::getCenter() const{
    return m_center;
}

Matrix4f Camera::getViewMatrix() const{
    return m_viewMatrix;
}

void Camera::perspective_projection(float angle,float aspect,float near,float far){
   
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    Matrix4f m=Matrix4f::perspectiveProjection(M_PI*angle/180.,aspect,near,far,false);
    glLoadMatrixf(m);
}
void Camera::planeTransform(Vector2f displacement){
    Vector3f x,y,z;
    createLocalFrame(m_eyepoint,m_center,m_up,x,y,z);

    m_eyepoint+=displacement.x()*x+displacement.y()*y;
    m_center=m_center+displacement.x()*x+displacement.y()*y;


    compute_viewMatrix();
} 
void Camera::rotateTransform(Vector2f displacement){
    Vector3f x,y,z;
    createLocalFrame(m_eyepoint,m_center,m_up,x,y,z);  

    Vector3f temp_eyepoint=m_eyepoint+displacement.x()*x+displacement.y()*y;
    Vector3f previous_y(y);
    createLocalFrame(temp_eyepoint,m_center,previous_y,x,y,z);

    float zoffset=(m_center-m_eyepoint).abs()-(m_center-temp_eyepoint).abs();

    m_eyepoint=temp_eyepoint+zoffset*z;
    m_up=y;
    compute_viewMatrix();
}
void Camera::compute_viewMatrix(){

    Vector3f x,y,z;
    createLocalFrame(m_eyepoint,m_center,m_up,x,y,z);

    
    m_viewMatrix.setCol(0,Vector4f(x,0));
    m_viewMatrix.setCol(1,Vector4f(y,0));
    m_viewMatrix.setCol(2,Vector4f(z,0));
    m_viewMatrix.setCol(3,Vector4f(m_eyepoint,1));
    m_viewMatrix=m_viewMatrix.inverse();
    
}

void Camera::mouseFunc(int button, int state,int x, int y){
    if(button==3||button==4){
        Vector3f d=(m_center-m_eyepoint).normalized();
        if(button==4) d=-d;
        setEyePoint(m_eyepoint+0.1*d);
    }
    
    if(button==GLUT_RIGHT_BUTTON){
        if(state==GLUT_DOWN){
            m_oper_type=CONST_MOVE;
            m_previous={x,y};
        }else if (state==GLUT_UP)
        {
            m_oper_type=CONST_NONE;
            m_previous={0,0};
        }   
    }

    if(button==GLUT_LEFT_BUTTON){
        if(state==GLUT_DOWN){
            m_oper_type=CONST_ROTATE;
            m_previous={x,y};
        }else if (state==GLUT_UP)
        {
            m_oper_type=CONST_NONE;
            m_previous={0,0};
        }   
    }
    glutPostRedisplay();
}
void Camera::screenVector2senceVector(Vector2f &displacement,float rate){
        displacement[0]*=-rate/m_screenW;
        displacement[1]*=rate/m_screenH;
}

void Camera::motionFunc(int x, int y){

    if(m_oper_type!=CONST_NONE){
        Vector2f current={x,y};
        Vector2f displacement=current-m_previous;
        m_previous=current;

        if(m_oper_type==CONST_MOVE){
            screenVector2senceVector(displacement);
            planeTransform(displacement);  
        }
        else if (m_oper_type==CONST_ROTATE){
            screenVector2senceVector(displacement,10);
            rotateTransform(displacement);
        }
            
        
        glutPostRedisplay();
    }

    
   
}