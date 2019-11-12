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

  RayTracer( SceneParser* scene, int max_bounces,float eps=1e-4);
  ~RayTracer();
  
  Vector3f traceRay( Ray& ray, float tmin, int bounces, 
                     float refr_index, Hit& hit ) const;
private:
  bool isShadows(const Vector3f& hitpoint,const Vector3f& light_dir,float light_distance) const;
  SceneParser* m_scene;
  Group * m_scence_group;
  int m_maxBounces;
  float m_eps;
};

#endif // RAY_TRACER_H
