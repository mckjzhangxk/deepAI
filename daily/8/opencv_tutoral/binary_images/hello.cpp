#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
int main(int argc, char* argv[]) {
    cv::Mat_<uint8_t> im = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (im.empty()) {
        std::cout << "can't open file" << std::endl;
        exit(0);
    }
    std::cout << "width:" << im.cols << ",height:" << im.rows << std::endl;
    cv::Mat_<uint8_t> dst;
    cv::threshold(im, dst, 128, 255, cv::ThresholdTypes::THRESH_BINARY);
    cv::imshow("window_name", dst);
    cv::waitKey();
}