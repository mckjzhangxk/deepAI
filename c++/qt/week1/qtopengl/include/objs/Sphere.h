#ifndef CAMERA_SPHERE
#define CAMERA_SPHERE
#include "Object3D.h"
#include "material.h"
#include<math.h>
class Sphere:public Object3D{
public:
    Sphere(int radius=0.2,int clips=20);
    void draw(bool);
    void setRadius(float radius);
    void setClips(float clips);
private:
    float m_radius;
    float m_clips;
    const float PI = 3.141592f;
    // Object3D interface
};
#endif
