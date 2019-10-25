#ifndef CAMERA_ZXK
#define CAMERA_ZXK
#include <GL/glut.h>
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

    void setEyePoint(Vector3f & eye);
    Vector3f getEyePoint() const;
    
    void setCenter(Vector3f & center);
    Vector3f getCenter() const;
    
    /**
     * 定义 透视投影的视椎，angle是y方向的视角，aspect是w/h，near,far定义了沿z轴的2个
     * 面，不再视椎内的物体不会被渲染出来
     * 
     * 3D到2D的转换有一下几个步骤：
     * 1.定义好视椎
     * 2.视椎 转化 成一个 长度为2个立方体，(-1<=x<=1,-1<=y<=1,-1<=z<=1),这个立方体称为image space
     * 3.在image space通过对z(深度)进行选择
    */
    void perspective_projection(float angle,float aspect,float near,float far);
private:
    Vector3f m_eyepoint;
    Vector3f m_pointTo;
    
    Vector3f m_x;
    Vector3f m_y;
    Vector3f m_z;
     
};
#endif