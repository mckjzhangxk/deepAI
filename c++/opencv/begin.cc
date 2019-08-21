/*
 *https://docs.opencv.org/3.4.6/db/df5/tutorial_linux_gcc_cmake.html
 * */
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc,char * argv[]){
 if (argc!=2){
 	cout<<"usage: DisplayImage.out <Image_Path>\n";
	return -1;
 }

 Mat img;

 img=imread(argv[1],1);
 if (!img.data){
    cout<<"No image data\n";
    return -1;
 }

 //namedWindow("Display Image",WINDOW_AUTOSIZE);
 imshow("DISPLAY",img);

 waitKey(0);

 return 0;
}
