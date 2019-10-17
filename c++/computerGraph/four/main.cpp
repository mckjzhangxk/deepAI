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

#include "bitmap_image.hpp"
int main( int argc, char* argv[] )
{
  // Fill in your implementation here.

  // This loop loops over each of the input arguments.
  // argNum is initialized to 1 because the first
  // "argument" provided to the program is actually the
  // name of the executable (in our case, "a4").
  char *infile;
  char *outfile;

  if(argc!=8){
    cout<<"usage:a4 -input infile -size 200 200 -output outfile"<<endl;
    exit(0);
  }
  assert (string("-input")==argv[1]);
  infile=argv[2];
  assert (string("-size")==argv[3]);
  int width=atoi(argv[4]);
  int height=atoi(argv[5]);
  assert(string("-output")==argv[6]);
  outfile=argv[7];
  // First, parse the scene using SceneParser.
  // Then loop over each pixel in the image, shooting a ray
  // through that pixel and finding its intersection with
  // the scene.  Write the color at the intersection to that
  // pixel in your output image.
  SceneParser scence(infile);
  Camera* camera=scence.getCamera();
  Group * objects=scence.getGroup();


  Image img(width,height);

  float wstep=2./width;
  float hstep=2./height;
  Vector3f defaultColor(0);
  for(int r=0;r<height;r++)
    for(int c=0;c<width;c++){
      float x=-1+r*hstep;
      float y=-1+c*wstep;
      Ray ray=camera->generateRay(Vector2f(x,y));
      Hit hit;
      bool isIntersect=objects->intersect(ray,hit,camera->getTMin());
      if(isIntersect){
      }
    }
  img.SaveImage(outfile);
  ///TODO: below demonstrates how to use the provided Image class
  ///Should be removed when you start
  // Vector3f pixelColor (1.0f,0,0);
  // //width and height
  // Image image( 10 , 15 );
  // image.SetPixel( 5,5, pixelColor );
  // image.SaveImage("demo.bmp");
  return 0;
}

