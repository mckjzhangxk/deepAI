#ifndef MATERIAL_H
#define MATERIAL_H

#include <cassert>
#include <vecmath.h>

#include "Ray.h"
#include "Hit.h"
#include "texture.hpp"
#include <math.h>
///TODO:
///Implement Shade function that uses ambient, diffuse, specular and texture
class Material
{
public:
	
 Material( const Vector3f& d_color ,const Vector3f& s_color=Vector3f::ZERO, float s=0):
  diffuseColor( d_color),specularColor(s_color), shininess(s)
  {
        	
  }

  virtual ~Material()
    {

    }

  virtual Vector3f getDiffuseColor() const 
  { 
    return  diffuseColor;
  }
    

  Vector3f Shade( const Ray& ray, const Hit& hit,
                  const Vector3f& dirToLight, const Vector3f& lightColor ) {
  
  
    //then get light direction,and normal
    Vector3f Ldir=dirToLight.normalized();
    Vector3f norm_dir=hit.getNormal();

    //L dot normal,clamp to 0
    float rate=max(0.f,Vector3f::dot(Ldir,norm_dir));
    Vector3f Kd=diffuseColor;
    if(t.valid()){
      Kd=t(hit.texCoord[0],hit.texCoord[1]);
    }
    Vector3f color=Kd*lightColor*rate;

    //specular color
    Vector3f ray_dir=ray.getDirection().normalized();
    Vector3f R=-Ldir+2*norm_dir*Vector3f::dot(norm_dir,Ldir);
    R.normalize();


    rate=pow(max(0.f,Vector3f::dot(R,-ray_dir)),shininess);
    
    Vector3f color1=lightColor*specularColor*rate;
    return color+color1 ; 
		
  }

  void loadTexture(const char * filename){
    t.load(filename);
  }
 protected:
  Vector3f diffuseColor;
  Vector3f specularColor;
  float shininess;
  Texture t;
};



#endif // MATERIAL_H
