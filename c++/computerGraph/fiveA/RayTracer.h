#ifndef RAY_TRACER_H
#define RAY_TRACER_H

#include <cassert>
#include <vector>
#include "SceneParser.h"
#include "Ray.h"
#include "Hit.h"

class SceneParser;


class RayTracer
{
public:
  
  RayTracer()
  {
      assert( false );
  }

  RayTracer( SceneParser* scene, int max_bounces,float eps=1e-2,bool shadows=false,bool reflection=false,bool refraction=false);
  ~RayTracer();
  
  Vector3f traceRay( Ray& ray, float tmin, int bounces, 
                     float refr_index, Hit& hit ) const;
  void setShadows(bool bl);
  void setReflection(bool bl);
  void setRefraction(bool bl);


private:
  bool isShadows(const Vector3f& hitpoint,const Vector3f& light_dir,float light_distance) const;
  SceneParser* m_scene;
  Group * m_scence_group;
  int m_maxBounces;
  float m_eps;
  bool m_show_shadows;
  bool m_show_reflection;
  bool m_show_refraction;
};

#endif // RAY_TRACER_H
