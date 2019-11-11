#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "SceneParser.h"
#include "Image.h"
#include "Camera.h"
#include <string.h>

using namespace std;

float clampedDepth ( float depthInput, float depthMin , float depthMax);

Vector3f get_pixel_color(const Ray & ray,const Hit & hit,const SceneParser & sence){
  Vector3f ambient_color=sence.getBackgroundColor()*sence.getAmbientLight();
  Vector3f diffuse_color(0);
  Material * material=hit.getMaterial();
  
  int numLight=sence.getNumLights();
  for(int i=0;i<numLight;i++){
    Light * light=sence.getLight(i);
    Vector3f lightColor;
    Vector3f light_direction;
    float dist;
    light->getIllumination(ray.pointAtParameter(hit.getT()),light_direction,lightColor,dist);
    
    diffuse_color+=material->Shade(ray,hit,light_direction,lightColor);
  }
  return ambient_color+diffuse_color;
}
#include "bitmap_image.hpp"
int main( int argc, char* argv[] )
{
  // Fill in your implementation here.

  // This loop loops over each of the input arguments.
  // argNum is initialized to 1 because the first
  // "argument" provided to the program is actually the
  // name of the executable (in our case, "a4").
  
  
  // char *p[14]={"./a4","-input","data/scene02_cube.txt","-size","200","200","-output","2.bmp","-normals","norm.bmp","-depth","8","12","depth.bmp"};
  // argv=p;
  // argc=14;

  char *infile;
  char *outfile;
  char *normfile=nullptr;
  char *depthfile=nullptr;
  double depth_low=0;
  double depth_high=0;
  if(argc<8){
    cout<<"usage:a4 -input infile -size 200 200 -output outfile [-normls normal_file.bmp]"<<endl;
    exit(0);
  }
  assert (string("-input")==argv[1]);
  infile=argv[2];
  assert (string("-size")==argv[3]);
  int width=atoi(argv[4]);
  int height=atoi(argv[5]);
  assert(string("-output")==argv[6]);
  outfile=argv[7];

 
  if(argc>=9){
    assert(string("-normals")==argv[8]);
    normfile=argv[9];
  }
 
  if(argc>=11){
    assert(string("-depth")==argv[10]);
    depth_low=atof(argv[11]);
    depth_high=atof(argv[12]);
    depthfile=argv[13];
  }
  
  // First, parse the scene using SceneParser.
  // Then loop over each pixel in the image, shooting a ray
  // through that pixel and finding its intersection with
  // the scene.  Write the color at the intersection to that
  // pixel in your output image.
  SceneParser scence(infile);
  Camera* camera=scence.getCamera();
  Group * objects=scence.getGroup();
  
  

  Image img(width,height);
  Image norm_img(width,height);
  Image depth_img(width,height);
 
  float wstep=2./width;
  float hstep=2./height;
  Vector3f defaultColor=scence.getBackgroundColor();
  //r=91,c=59
  for(int r=0;r<height;r++)
    for(int c=0;c<width;c++){
      if(r==91 && c==59){
        Vector3f(3);
      }
      float x=-1+c*wstep;
      float y=-1+r*hstep;

      Ray ray=camera->generateRay(Vector2f(x,y));
      Hit hit;
      bool isIntersect=objects->intersect(ray,hit,camera->getTMin());
      if(isIntersect){
        if(normfile){
          Vector3f hitNorm=hit.getNormal();
          Vector3f color_norm(abs(hitNorm.x()),abs(hitNorm.y()),abs(hitNorm.z()));
          norm_img.SetPixel(c,r,color_norm);     
        }
        if(depthfile){
          float ratio=(depth_high-hit.getT())/(depth_high-depth_low);
          ratio=max(ratio,0.f);
          ratio=min(ratio,1.f);
          Vector3f color_depth=Vector3f(1,1,1)*ratio;
          depth_img.SetPixel(c,r,color_depth);     
        }
        Vector3f pixel=get_pixel_color(ray,hit,scence);
        
        img.SetPixel(c,r,pixel);
      }else{
        img.SetPixel(c,r,defaultColor);
      }

    }
  img.SaveImage(outfile);
  if(normfile)
    norm_img.SaveImage(normfile);
  if(depthfile)
    depth_img.SaveBMP(depthfile);

  ///TODO: below demonstrates how to use the provided Image class
  ///Should be removed when you start
  // Vector3f pixelColor (1.0f,0,0);
  // //width and height
  // Image image( 10 , 15 );
  // image.SetPixel( 5,5, pixelColor );
  // image.SaveImage("demo.bmp");
  return 0;
}

