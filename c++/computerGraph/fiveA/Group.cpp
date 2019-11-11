#include "Group.h"

bool Group::intersect(const Ray &ray,Hit &h,float tmin){
    bool ret=false;
    for(Object3D* obj:this->mObjects){
        if(obj->intersect(ray,h,tmin))
            ret=true;
    }
    return ret;
}