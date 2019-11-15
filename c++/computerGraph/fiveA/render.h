#ifndef RENDER_H
#define RENDER_H
#include "Image.h"
#include "RayTracer.h"
#include "Camera.h"

class Render
{
public:

    Render(int w=300,int h=300,bool jitter=false,bool filter=false,int jitter_scale=3,float jitter_dev=0.5);
    void setDimension(int w,int h);
    void setJitter(bool bl=true);
    void setFilter(bool bl=true);
    void setJitter_dev(float b);
    Image* run(RayTracer* rayTracer,Camera * camera);

private:
    Image* gaussFilter(const Image & I);
    Image* downSample(const Image & I,int stride=3);
    int m_w;
    int m_h;
    bool m_jitter;


    bool m_filter;
    int m_jitter_scale;
    float m_jitter_dev;
};

#endif // RENDER_H
