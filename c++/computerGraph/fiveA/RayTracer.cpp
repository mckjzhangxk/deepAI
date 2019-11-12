#include "RayTracer.h"
#include "Camera.h"
#include "Ray.h"
#include "Hit.h"
#include "Group.h"
#include "Material.h"
#include "Light.h"

#define EPSILON 0.001

//IMPLEMENT THESE FUNCTIONS
//These function definitions are mere suggestions. Change them as you like.
Vector3f mirrorDirection( const Vector3f& normal, const Vector3f& incoming)
{
    Vector3f k=(Vector3f::dot(incoming,normal)/normal.absSquared())*normal;
    return incoming-2*k;
}

bool transmittedDirection( const Vector3f& normal, const Vector3f& incoming, 
        float index_n, float index_nt, 
        Vector3f& transmitted)
{

}

RayTracer::RayTracer( SceneParser * scene, int max_bounces,float eps) :
  m_scene(scene),m_eps(eps)

{
  m_scence_group=scene->getGroup();
  m_maxBounces = max_bounces;
}

RayTracer::~RayTracer()
{
}

Vector3f RayTracer::traceRay( Ray& ray, float tmin, int bounces,
        float refr_index, Hit& hit ) const
{
    if(bounces>=m_maxBounces)
      return Vector3f(0.);

    hit = Hit( FLT_MAX, NULL, Vector3f( 0, 0, 0 ) );

    
    Vector3f diffColor(0.);
    if(m_scence_group->intersect(ray,hit,tmin)){
      Vector3f hitpoint=ray.pointAtParameter(hit.getT());
      Material * hitMaterial=hit.getMaterial();
      int lightNum=m_scene->getNumLights();
      for(int l=0;l<lightNum;l++){
        Light* light=m_scene->getLight(l);
        Vector3f light_dir,light_color;
        float light_distance;
        light->getIllumination(hitpoint,light_dir,light_color,light_distance);
        if(!isShadows(hitpoint,light_dir,light_distance)){
            diffColor+=hitMaterial->Shade(ray,hit,light_dir,light_color);
        }

      }
      Vector3f returnColor=diffColor;
      //secondary ray!
       Vector3f outcoming=mirrorDirection(hit.getNormal(),ray.getDirection());
       Ray sec_ray(hitpoint,outcoming);
       returnColor+=diffColor*traceRay(sec_ray,tmin,bounces+1,refr_index,hit);
       //simple refraction
       hitMaterial->getRefractionIndex();

       return returnColor;
    }else{
        return m_scene->getBackgroundColor(ray.getDirection());
    }
}

bool RayTracer::isShadows(const Vector3f &hitpoint, const Vector3f &light_dir, float light_distance) const
{
    Ray ray(hitpoint,light_dir.normalized());
    Hit hit(FLT_MAX, nullptr, Vector3f( 0.f));

    bool flag=m_scence_group->intersect(ray,hit,m_eps);
    if(flag&&hit.getT()+m_eps<light_distance){
        return true;
    }
    return false;
}
