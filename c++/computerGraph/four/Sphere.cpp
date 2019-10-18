#include "Sphere.h"
#include <math.h>

void quadratic_solve(float a,float b,float c,bool &solved,float & x1,float &x2){
    float delta=b*b-4*a*c;
    if(delta<0){
        solved=false;
    }else{
        solved=true;
        float t1=(-b+sqrt(delta))/(2*a);
        float t2=(-b-sqrt(delta))/(2*a);
        x1=min(t1,t2);
        x2=max(t1,t2);
    }
    
}
bool Sphere::intersect(const Ray &ray,Hit &h, float tmin){

    Vector3f light_orgin=ray.getOrigin();
    Vector3f light_dir=ray.getDirection();
    
    float a=Vector3f::dot(light_dir,light_dir);
    float b=2*Vector3f::dot(light_dir,light_orgin-center);
    float c=Vector3f::dot(light_orgin-center,light_orgin-center)-radius*radius;
    bool isSolved=false;
    float t1,t2;
    quadratic_solve(a,b,c,isSolved,t1,t2);

    if(isSolved && t1>tmin && t1<h.getT()){
        Vector3f norm=(ray.pointAtParameter(t1)-center).normalized();
        h.set(t1,material,norm);
    }
    return isSolved==true &&t1>tmin;    
}