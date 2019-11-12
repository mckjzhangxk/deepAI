#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vecmath.h>
#include "Object3D.h"
///TODO implement this class
///So that the intersect function first transforms the ray
///Add more fields as necessary
class Transform: public Object3D
{
public: 
  Transform(){}
 Transform( const Matrix4f& m, Object3D* obj ):o(obj){
    Matrix4f xx=m;
    xx.print();
  
    m_2local=m.inverse();
    m_2localT=m_2local.transposed();
  }
  ~Transform(){
  }
  virtual bool intersect( const Ray& r , Hit& h , float tmin){


    Vector4f r_org(r.getOrigin(),1);
    Vector4f r_dir(r.getDirection(),0);

    Vector3f r_org_new=(m_2local*r_org).xyz();
    Vector3f r_dir_new=(m_2local*r_dir).xyz();

    Ray rnew(r_org_new,r_dir_new);

    bool flag=o->intersect( rnew , h , tmin);
    if(flag){
      Vector4f norm(h.getNormal(),0);
      Vector3f newNorm=(m_2localT*norm).xyz().normalized();
      h.set(h.getT(),h.getMaterial(),newNorm);
    }
    return flag;
  }

 protected:
  Object3D* o; //un-transformed object	
  Matrix4f m_2local;
  Matrix4f m_2localT;
};

#endif //TRANSFORM_H
