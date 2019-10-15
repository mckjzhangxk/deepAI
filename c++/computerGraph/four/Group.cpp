#include "Group.h"

bool Group::intersect(const Ray &ray,Hit &h,float tmin){
    for(Object3D* obj:this->mObjects){
        obj->intersect(ray,h,tmin);
    }
}