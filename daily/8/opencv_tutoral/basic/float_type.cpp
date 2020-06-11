#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

/**
 * 保存浮点的时候最好使用.exr格式,这样会原样保存
 * 尝试把exr改成png看看会如何
*/
int main() {
    using Matf = cv::Mat_<float>;
    Matf I(10, 10);
    I.at<float>(5, 5) = 32.32;
    const char *filename = "test.exr";
    cv::imwrite(filename, I);

    Matf I1 = cv::imread(filename, cv::IMREAD_UNCHANGED);
    std::cout << "I=" << I.at<float>(5, 5) << std::endl;
    std::cout << "I1=" << I1.at<float>(5, 5) << std::endl;
}