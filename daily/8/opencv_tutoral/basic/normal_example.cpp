#include "cmath"
#include "common.h"
#include "iostream"
using namespace cv;
using namespace std;
/**
 * cv::normalize	(	InputArray 	src,
            InputOutputArray 	dst,
            double 	alpha = 1,
            double 	beta = 0,
            int 	norm_type = NORM_L2,
            int 	dtype = -1,
            InputArray 	mask = noArray() 
            )	
 * 
 * 当L1,L2,Linf的时候alpha有用,把alpha设置成1就可以
 * L_min_max的时候,最小是alpha,最大是beta,默认是(0,1)
 * y=(x-min)/(max-min)
*/
int main() {
    cv::Mat_<float> m(2, 8);
    cv::Mat_<float> ml(2, 8);

    float dist_L2 = 0;
    float dist_L1 = 0;
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++) {
            float e = rand() & 0x3ff;
            dist_L2 += e * e;
            dist_L1 += abs(e);
            m(i, j) = e;
        }
    cout << "L1" << endl;
    cout << dist_L1 << endl;
    cout << cv::norm(m, cv::NORM_L1) << endl;

    cout << "L2" << endl;
    cout << sqrt(dist_L2) << endl;
    cout << cv::norm(m, cv::NORM_L2) << endl;

    cv::Mat_<float> m2(1, 3);
    cv::Mat result;
    m2(0, 0) = 3;
    m2(0, 1) = 4;
    m2(0, 2) = 0;
    //L1=12,结果是 3/7,4/7,0/7
    normalize(m2, result, 1, 22, NORM_L1);
    cout << result << endl;
    //L2=5 ,结果是 3/5,4/5,0/5
    normalize(m2, result, 33, 22, NORM_L2);
    cout << result << endl;
    //Linf=4,结果是 3/4,4/4
    normalize(m2, result, 1, 22, NORM_INF);
    cout << result << endl;

    //(x-min)/(max-min)
    // normalize(m2, result, 0, 1, NORM_MINMAX);
    // cout << result << endl;
}