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
#include "render.h";

using namespace std;

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
  
  
   char *p[]={"./a5","-input","scene10_sphere.txt","-size","300","300","-output","6.bmp","-bounces","2"};
   argv=p;
   argc=10;
  Parse_Result args=parse_input(argc,argv);


  
  // First, parse the scene using SceneParser.
  // Then loop over each pixel in the image, shooting a ray
  // through that pixel and finding its intersection with
  // the scene.  Write the color at the intersection to that
  // pixel in your output image.
  SceneParser scence(args.infile.c_str());
  RayTracer rayTracer(&scence,args.maxBounce);
  rayTracer.setShadows(true);
  rayTracer.setReflection(true);
  rayTracer.setRefraction(false);
  Camera* camera=scence.getCamera();


  Render render(args.width,args.height,false,false,3,0.5);
  render.setJitter(true);
  render.setFilter(true);

  Image* img=render.run(&rayTracer,camera);


  img->SaveImage(args.outfile.c_str());



  return 0;
}

