#ifndef PLANE_H
#define PLANE_H

#include "Object3D.h"
#include <vecmath.h>
#include <cmath>
using namespace std;
///TODO: Implement Plane representing an infinite plane
///choose your representation , add more fields and fill in the functions
class Plane: public Object3D
{
public:
	Plane(){}
	Plane( const Vector3f& normal , float d , Material* m):Object3D(m){
		m_d=d;
		m_normal=normal;
		m_matrial=m;
	}
	~Plane(){}
	virtual bool intersect( const Ray& r , Hit& h , float tmin){
		
	}

protected:
	Material *m_matrial;
	Vector3f m_normal;
	float m_d;
};
#endif //PLANE_H
		

