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
}

bool transmittedDirection( const Vector3f& normal, const Vector3f& incoming, 
        float index_n, float index_nt, 
        Vector3f& transmitted)
{
}

RayTracer::RayTracer( SceneParser * scene, int max_bounces) :
  m_scene(scene)

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
      return Vector3f(0,0,0);

    hit = Hit( FLT_MAX, NULL, Vector3f( 0, 0, 0 ) );
    if(bounces>=m_maxBounces)
      return Vector3f(0,0,0);
    
    Vector3f diffColor(0.0f);
    if(m_scence_group->intersect(ray,hit,tmin)){
      Vector3f hitpoint=ray.pointAtParameter(hit.getT());
      Material * hitMaterial=hit.getMaterial();
      int lightNum=m_scene->getNumLights();
      for(int l=0;l<lightNum;l++){
        Light* light=m_scene->getLight(l);
        Vector3f light_dir,light_color;
        float light_distance;
        light->getIllumination(hitpoint,light_dir,light_color,light_distance);

        diffColor+=hitMaterial->Shade(ray,hit,light_dir,light_color);
      }

      //secondary ray!
      Vector3f outcoming=mirrorDirection(hit.getNormal(),ray.getDirection());
      Ray sec_ray(hitpoint,outcoming);
      diffColor+=traceRay(sec_ray,tmin,bounces+1,refr_index,hit);
    }
    return diffColor;
}
