#include "Camera.h"


Ray PerspectiveCamera::generateRay(const Vector2f& point){
    float D=1;

    float H=D*tan(m_angle/2);
    float W=H*1.0;


    float x=point[0];
    float y=point[1];

    Vector3f ray_dir=x*W*horizontal+
                     y*H*up+
                     1*D*direction;
    ray_dir.normalize();

    Ray ray(center,ray_dir);
    return ray;
}
