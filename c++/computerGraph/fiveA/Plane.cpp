#include "Plane.h"


bool Plane::intersect(const Ray &ray,Hit &hit,float tmin){
    float r=Vector3f::dot(m_normal,ray.getDirection());

    if(abs(r)<1e-4)
        return false;
    
    float t=-(m_d+Vector3f::dot(m_normal,ray.getOrigin())) /r;
    if(t>tmin&&t<hit.getT()){
        hit.set(t,m_matrial,m_normal.normalized());
        return true;
    }else
    {
        return false;
    }
    
}