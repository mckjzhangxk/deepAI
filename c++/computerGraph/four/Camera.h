#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"
#include <vecmath.h>
#include <float.h>
#include <cmath>


class Camera
{
public:
	//generate rays for each screen-space coordinate
	virtual Ray generateRay( const Vector2f& point ) = 0 ; 
	
	virtual float getTMin() const = 0 ; 
	virtual ~Camera(){}
protected:
	Vector3f center; 
	Vector3f direction;
	Vector3f up;
	Vector3f horizontal;

};

///TODO: Implement Perspective camera
///Fill in functions and add more fields if necessary
class PerspectiveCamera: public Camera
{
public:
	PerspectiveCamera(const Vector3f& center, const Vector3f& direction,const Vector3f& up , float angle){
			Vector3f w=direction.normalized();
			Vector3f u=Vector3f::cross(w,up).normalized();
			Vector3f v=Vector3f::cross(u,w).normalized();

			m_u=u;m_v=v;m_w=w;
			m_e=center;
			m_angle=angle;
	}

	virtual Ray generateRay( const Vector2f& point){
		
	}

	virtual float getTMin() const { 
		return 0.0f;
	}

private:
	Vector3f m_u;
	Vector3f m_v;
	Vector3f m_w;
	Vector3f m_e;
	float m_angle;
};

#endif //CAMERA_H
