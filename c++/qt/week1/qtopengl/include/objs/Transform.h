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
    m_2world=m;
    m_2local=m.inverse();
    m_2localT=m_2local.transposed();
  }
  ~Transform(){
  }

 protected:
  Object3D* o; //un-transformed object	
  Matrix4f m_2world;
  Matrix4f m_2local;
  Matrix4f m_2localT;

  // Object3D interface
public:
  void draw(bool);
};

#endif //TRANSFORM_H
