#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>

#include <iostream>

using namespace cv;
using namespace std;


int main(int argc,char* argv[]){

String imagename;

if (argc>1){
imagename=argv[1];
}

Mat img;
cout<<imagename<<endl;
img=imread(imagename,IMREAD_COLOR);
if(img.empty()){
    cout <<  "Could not open or find the image" << std::endl ;
    return -1;
}
imshow("Display",img);

waitKey(0);

return 0;

}
