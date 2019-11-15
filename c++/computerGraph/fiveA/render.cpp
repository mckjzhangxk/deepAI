#include "render.h"

float rangeRandom(float delta){
    float min = -delta;
    float max = delta;
    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}
Render::Render(int w, int h, bool jitter, bool filter, int jitter_scale,float jitter_dev):
    m_w(w),m_h(h),m_jitter(jitter),m_filter(filter),m_jitter_scale(jitter_scale),m_jitter_dev(jitter_dev)
{

}

void Render::setDimension(int w, int h)
{
    m_w=w;
    m_h=h;
}

void Render::setJitter(bool bl)
{
    m_jitter=bl;
}

void Render::setFilter(bool bl)
{
    m_filter=bl;
}

void Render::setJitter_dev(float b)
{
    m_jitter_dev=b;

}

Image* Render::run(RayTracer *rayTracer, Camera *camera)
{
    int H=m_jitter?m_jitter_scale*m_h:m_h;
    int W=m_jitter?m_jitter_scale*m_w:m_w;

    Image* img=new Image(W,H);

    float wstep=2./W;
    float hstep=2./H;

    for(int r=0;r<H;r++)
      for(int c=0;c<W;c++){
        float x=-1+c*wstep;
        float y=-1+r*hstep;

        if(m_jitter){
            x+=rangeRandom(m_jitter_dev)*wstep;
            y+=rangeRandom(m_jitter_dev)*hstep;
        }
        Ray ray=camera->generateRay(Vector2f(x,y));

        Hit hit;
        Vector3f pixel=rayTracer->traceRay(ray,0,0,1,hit);
        img->SetPixel(c,r,pixel);
      }

    if(m_jitter&&m_filter){
        Image* filter_image=gaussFilter(*img);
        Image* dowmsample_image=downSample(*filter_image);
        delete img;delete filter_image;
        return dowmsample_image;
    }if(m_filter){
        Image* filter_image=gaussFilter(*img);
        delete img;
        return filter_image;
    }else{
        return img;
    }
}

Image* Render::gaussFilter(const Image &I)
{
    float kernel[]={0.1201,0.2339,0.2931,0.2339,0.1201};
    Image* img=new Image(I.Width(),I.Height());
    for(int r=0;r<I.Height();r++)
        for(int c=0;c<I.Width();c++){
            Vector3f rf;
            for(int k=-2;k<=2;k++){
                int C=min(max(k+c,0),I.Width()-1);
                Vector3f x=I.GetPixel(r,C);
                rf+=x*kernel[k];
            }
            img->SetPixel(r,c,rf);
        }


    for(int c=0;c<I.Width();c++)
        for(int r=0;r<I.Height();r++){
            Vector3f rf;
            for(int k=-2;k<=2;k++){
                int R=min(max(k+r,0),I.Height()-1);
                Vector3f x=I.GetPixel(R,c);
                rf+=x*kernel[k];
            }
            img->SetPixel(r,c,rf);
        }

    return img;
}

Image* Render::downSample(const Image &I, int stride)
{
    int H=I.Height()/stride;
    int W=I.Width()/stride;

    Image* img=new Image(W,H);

    for(int r=0;r<H;r++)
      for(int c=0;c<W;c++){
        Vector3f pixel;
        for(int kh=0;kh<stride;kh++)
            for(int kw=0;kw<stride;kw++){
                pixel+=I.GetPixel(stride*r+kh,stride*c+kw);
            }
            pixel=pixel/9.;
            img->SetPixel(r,c,pixel);
      }

    return img;
}
