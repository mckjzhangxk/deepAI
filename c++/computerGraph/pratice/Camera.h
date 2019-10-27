#ifndef CAMERA_ZXK
#define CAMERA_ZXK
#include <GL/glut.h>
#include <Vector2f.h>
#include <Vector3f.h>
#include<Vector4f.h>
#include<Matrix4f.h>
#include<iostream>
#include<cmath>


using namespace std;
class Camera{
public:
    /**
     * 建立一个眼点”右手“坐标系，z的方向 是 ”看向“ 的反方向，
     * 然后计算x垂直与z和传入的up,最后计算y.x,y,z都是单位向量
     * 
    */
    Camera();
    Camera(Vector3f,Vector3f,Vector3f);
    /**
     * glutLookAt过于神秘，其实它设置了
     * 一个把 世界坐标  转成 眼点 坐标 的矩阵R_world_2_local,
     * 
     * 为什么需要这个R_world_2_local?
     * 有一个绝对的坐标系统，它根据R_world_2_local,把场景的每个
     * 物体在这个坐标系统重画，然后沿着-Z的方向生成视椎(perspective_projection)，
     * 视椎内的物体被渲染显示
    */
    Matrix4f getViewMatrix() const;

    void setDimension(int W,int H);
    void setEyePoint(Vector3f & eye);
    void setEyePoint(Vector3f && eye);
    Vector3f getEyePoint() const;
    
    void setCenter(Vector3f & center);
    void setCenter(Vector3f && center);
    Vector3f getCenter() const;
    
    /**
     * 定义 透视投影的视椎，angle是y方向的视角，aspect是w/h，near,far定义了沿z轴的2个
     * 面，不再视椎内的物体不会被渲染出来
     * 
     * 3D到2D的转换有一下几个步骤：
     * 1.定义好视椎
     * 2.视椎 转化 成一个 长度为2个立方体，(-1<=x<=1,-1<=y<=1,-1<=z<=1),这个立方体称为image space
     * 3.在image space通过对z(深度)进行选择
     * 
     * perspective_projection其实是定义了1.2 步骤
    */
    void perspective_projection(float angle,float aspect,float near,float far);

    void planeTransform(Vector2f displace);
    void rotateTransform(Vector2f displace);

    void mouseFunc(int button, int state,int x, int y);
    void motionFunc(int x, int y);
private:
    void compute_viewMatrix();
    /**
     * 鼠标划过形成的 vector 的坐标 
     * 转换成view frame{x,y}方向的坐标,
     * 需要m_screenW,m_screenH的正确设置
    */
    void screenVector2senceVector(Vector2f & v,float rate=2);
    Vector3f m_eyepoint;
    Vector3f m_center;
    Vector3f m_up;
    Matrix4f m_viewMatrix;   

    float m_screenW;
    float m_screenH;
    //鼠标相关
    Vector2f m_previous;
    int m_oper_type;

    static const int CONST_NONE=0;
    static const int CONST_MOVE=1;
    static const int CONST_ROTATE=2;

};
#endif