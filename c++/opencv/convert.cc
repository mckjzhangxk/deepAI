#include <opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;
int main(int argc,char** argv){

	Mat img;
	img=imread(argv[1],IMREAD_COLOR);
	if(img.empty()){
		cout<<"empty image"<<endl;
		return 0;	
	}
	Mat gray_img;
	cvtColor(img,gray_img,COLOR_BGR2GRAY);
	imwrite("hello.jpg",gray_img);

	imshow("orgin",img);
	imshow("gray",gray_img);

	waitKey(0);

}
