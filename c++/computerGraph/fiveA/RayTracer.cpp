#include "RayTracer.h"
#include "Camera.h"
#include "Ray.h"
#include "Hit.h"
#include "Group.h"
#include "Material.h"
#include "Light.h"

#define EPSILON 0.001
Vector3f blendColor(float index_n,float index_nt,const Vector3f& refColor,const Vector3f& refracColor,
                    const Vector3f& n,const Vector3f& d,const Vector3f& t ){

    float R0=pow((index_n-index_nt)/(index_n+index_nt),2);
    float c;

    if(index_n<=index_nt)
        c=abs(Vector3f::dot(n.normalized(),d.normalized()));
    else
        c=abs(Vector3f::dot(n.normalized(),t.normalized()));
    float R=R0+(1-R0)*pow(1-c,5);

    return R*refColor+(1-R)*refracColor;
}
//IMPLEMENT THESE FUNCTIONS
//These function definitions are mere suggestions. Change them as you like.
Vector3f mirrorDirection( const Vector3f& normal, const Vector3f& incoming)
{
    Vector3f N=normal.normalized();
    Vector3f I=incoming.normalized();

    float dot=Vector3f::dot(N,I);
//    Vector3f k=dot*N;
//    return -I+2*k;
    Vector3f k=-dot*N;
    return I+2*k;
}

bool transmittedDirection( const Vector3f& normal, const Vector3f& incoming, 
        float index_n, float index_nt, 
        Vector3f& transmitted)
{
    Vector3f N=normal.normalized();
    Vector3f I=incoming.normalized();
    float eta=index_n/index_nt;

    float dot=Vector3f::dot(N,I);

    float delta=1-pow(eta,2)*(1-pow(dot,2));
    if(delta<0)
        return false;

    Vector3f x1=-sqrt(delta)*N;
    Vector3f x2=eta*(I-dot*N);
//    transmitted=(-sqrt(delta)*N+eta*(I-dot*N));
    transmitted=x1+x2;
    return true;
}

RayTracer::RayTracer( SceneParser * scene, int max_bounces,float eps,bool shadows,bool reflection,bool refraction) :
  m_scene(scene),m_eps(eps),m_show_shadows(shadows),m_show_reflection(reflection),m_show_refraction(refraction)

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


    if(m_scence_group->intersect(ray,hit,tmin)){
      bool outside=(Vector3f::dot(hit.getNormal(),ray.getDirection())<0);

      Vector3f diffColor(0.);
      //compute diffcoior color
      Vector3f hitpoint=ray.pointAtParameter(hit.getT());
      Material * hitMaterial=hit.getMaterial();
      int lightNum=m_scene->getNumLights();
      for(int l=0;l<lightNum;l++){
        Light* light=m_scene->getLight(l);
        Vector3f light_dir,light_color;
        float light_distance;
        light->getIllumination(hitpoint,light_dir,light_color,light_distance);
        if(!m_show_shadows||!isShadows(hitpoint,light_dir,light_distance)){
            diffColor+=hitMaterial->Shade(ray,hit,light_dir,light_color);
        }

      }


      bool canRecursive=bounces<m_maxBounces;
      //secondary ray!
       Vector3f returnColor(diffColor);
       Vector3f reflectColor(canRecursive*hitMaterial->getSpecularColor());
       Vector3f reflectionDirection(0.f);
       bool haveReflect=m_show_reflection&canRecursive&(reflectColor!=Vector3f::ZERO);

       Vector3f refractionColor(0.f);
       Vector3f refractDirection(0);
       float index_nt=hitMaterial->getRefractionIndex();


       bool haveRefract=m_show_refraction&canRecursive&(index_nt>0);
       //simple reflection
       if(haveReflect){
           reflectionDirection=mirrorDirection(hit.getNormal(),ray.getDirection());
           Ray reflectRay(hitpoint,reflectionDirection);
           Hit hit1;
           reflectColor=reflectColor*traceRay(reflectRay,m_eps,bounces+1,refr_index,hit1);
       }
       //simple refraction
       if(haveRefract){
            if(!outside&&refr_index==index_nt)
                index_nt=1.0;
            haveRefract=transmittedDirection(outside?hit.getNormal():-hit.getNormal(),ray.getDirection(),refr_index,index_nt,refractDirection);
            if(haveRefract){
                    Ray refractRay(hitpoint,refractDirection);
                    Hit hit2;
                    refractionColor=traceRay(refractRay,m_eps,bounces+1,index_nt,hit2);
            }

       }


       if(haveReflect&&haveRefract){
            returnColor+=blendColor(refr_index,index_nt,reflectColor,refractionColor,hit.getNormal(),reflectionDirection,refractDirection);
       }else if(haveReflect){
           returnColor+=reflectColor;
       }else if(haveRefract){
           returnColor+=refractionColor;
       }
     return returnColor;

    }else{
        return m_scene->getBackgroundColor(ray.getDirection());
    }
}

void RayTracer::setShadows(bool bl)
{
    m_show_shadows=bl;
}

void RayTracer::setReflection(bool bl)
{
    m_show_reflection=bl;
}

void RayTracer::setRefraction(bool bl)
{
    m_show_refraction=bl;
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
