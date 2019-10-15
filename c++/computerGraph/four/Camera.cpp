#include "Camera.h"


Ray PerspectiveCamera::generateRay(const Vector2f& point){
    float D=1;

    float R=D*tan(m_angle/2);

    float x=point[0];
    float y=point[1];

    Vector3f ray_dir=x*R*m_u+y*R*m_v+D*m_w+m_e;
    ray_dir.normalize();

    Ray ray(m_e,ray_dir);
    return ray;
}
