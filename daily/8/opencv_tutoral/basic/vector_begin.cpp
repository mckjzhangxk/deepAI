#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

/**
 * 
 * opencv matrix表示矩阵,但M(i,j)可以不是基本类型
*/
int main(int argc, char* argv[]) {
    using Vec = cv::Vec3b;
    if (argc > 1) {
        cv::Mat_<Vec> I = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (I.empty()) {
            std::cout << "can't open file" << std::endl;
            exit(0);
        }

        char* windowname = "zxk";
        cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowname, I);
        cv::waitKey();

        for (int i = 20; i < 200; i++)
            for (int j = 20; j < 200; j++) {
                I.at<Vec>(i, j) = {0, 0, 0};
            }
        cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowname, I);
        cv::waitKey();
    }
}