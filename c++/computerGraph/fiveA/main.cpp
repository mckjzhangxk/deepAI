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
#include "RayTracer.h"
#include "bitmap_image.hpp"
using namespace std;

float clampedDepth ( float depthInput, float depthMin , float depthMax);

Vector3f get_pixel_color(const Ray & ray,const Hit & hit,const SceneParser & sence){
  Vector3f ambient_color=sence.getBackgroundColor(Vector3f(0))*sence.getAmbientLight();
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

struct Parse_Result
{
  string infile;
  string outfile;
  int width;
  int height;
  int maxBounce;
};

Parse_Result parse_input(int argc, char* argv[]){
  Parse_Result ret;

  if(argc<8){
    cout<<"usage:a5 -input infile -size 200 200 -output outfile"<<endl;
    exit(0);
  }
  assert (string("-input")==argv[1]);
  ret.infile=argv[2];
  assert (string("-size")==argv[3]);
  ret.width=atoi(argv[4]);
  ret.height=atoi(argv[5]);
  assert(string("-output")==argv[6]);
  ret.outfile=argv[7];

  if(argc>8)
  assert(string("-bounces")==argv[8]);
  ret.maxBounce=atoi(argv[9]);


  return ret;
}

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
  Parse_Result args=parse_input(argc,argv);


  
  // First, parse the scene using SceneParser.
  // Then loop over each pixel in the image, shooting a ray
  // through that pixel and finding its intersection with
  // the scene.  Write the color at the intersection to that
  // pixel in your output image.
  SceneParser scence(args.infile.c_str());
  RayTracer rayTracer(&scence,args.maxBounce);
  Camera* camera=scence.getCamera();

  Image img(args.width,args.height);

 
  float wstep=2./args.width;
  float hstep=2./args.height;
  Vector3f defaultColor=scence.getBackgroundColor(Vector3f());
  for(int r=0;r<args.height;r++)
    for(int c=0;c<args.width;c++){
      float x=-1+c*wstep;
      float y=-1+r*hstep;

      Ray ray=camera->generateRay(Vector2f(x,y));

      
      Hit hit;
      Vector3f pixel=rayTracer.traceRay(ray,0,-1,0,hit);
      img.SetPixel(c,r,pixel);
    }
  img.SaveImage(args.outfile.c_str());


  ///TODO: below demonstrates how to use the provided Image class
  ///Should be removed when you start
  // Vector3f pixelColor (1.0f,0,0);
  // //width and height
  // Image image( 10 , 15 );
  // image.SetPixel( 5,5, pixelColor );
  // image.SaveImage("demo.bmp");
  return 0;
}

