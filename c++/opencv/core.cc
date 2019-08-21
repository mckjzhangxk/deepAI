#include<opencv2/opencv.hpp>


using namespace cv;


int main(int argc,char** argv){

 Mat img;
 img=imread(argv[1],1);

 Mat img1(img,Rect(10,10,500,500));

 Mat img2=img(Range::all(),Range(100,300));
 imshow("clip",img1);
 imshow("clip",img2);
 waitKey(0);
 waitKey(0);
}

