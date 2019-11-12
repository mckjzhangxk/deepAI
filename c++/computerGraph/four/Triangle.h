#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Object3D.h"
#include <vecmath.h>
#include <cmath>
#include <iostream>

using namespace std;
///TODO: implement this class.
///Add more fields as necessary,
///but do not remove hasTex, normals or texCoords
///they are filled in by other components
class Triangle: public Object3D
{
public:
	Triangle();
        ///@param a b c are three vertex positions of the triangle
	Triangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, Material* m):Object3D(m){
          hasTex = false;

		  m_a=a;m_b=b;m_c=c;
		//   m_normal=Vector3f::cross(b-a,c-a).normalized();
		  m_material=m;
	}
	Vector3f interpNorm(float a,float b,float c){
		return (a*normals[0]+b*normals[1]+c*normals[2]).normalized();
	}
	
	virtual bool intersect( const Ray& ray,  Hit& hit , float tmin){
		Vector3f eyepoint=ray.getOrigin();
		Vector3f d=ray.getDirection();

	 	Matrix3f m(m_b-m_a,m_c-m_a,-d);
		Vector3f f=eyepoint-m_a;
		
		bool sigular;
        Matrix3f minv=m.inverse(&sigular,1e-6);
		if(!sigular){
			Vector3f result=minv*f;
			float r2=result[0],r3=result[1],t=result[2];
			if(r2>=0&&r3>=0&&((r2+r3)<=1)&&t>tmin&&t<hit.getT()){
					hit.set(t,m_material,interpNorm(1-r2-r3,r2,r3));
					if(hasTex){
						Vector2f texcoord=texCoords[0]*(1-r2-r3)+texCoords[1]*r2+texCoords[2]*r3;
						hit.setTexCoord(texcoord);
					}
				return true;
			}else
			{
				return false;
			}
			

		}else{
			return false;
		}
		
	}
	bool hasTex;
	Vector3f normals[3];
	Vector2f texCoords[3];
protected:
	Material* m_material;
	Vector3f m_a;
	Vector3f m_b;
	Vector3f m_c;
	// Vector3f m_normal;
};

#endif //TRIANGLE_H
