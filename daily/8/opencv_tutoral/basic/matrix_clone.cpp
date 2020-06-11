#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

/**
 * 本例展示mat copy不是复制内容,而是像smart_point一样
 * 把引用给别的对象
*/
int main() {
    using Matf = cv::Mat_<float>;
    Matf I(10, 10);
    Matf I_nocopy = I;
    Matf I_copy = I.clone();

    std::cout << "I=" << I.at<float>(3, 3) << std::endl;

    I_nocopy.at<float>(3, 3) = 27;
    std::cout << "after change I_nocopy:I=" << I.at<float>(3, 3) << std::endl;

    I_copy.at<float>(3, 3) = 82;
    std::cout << "after change I_copy:I=" << I.at<float>(3, 3) << std::endl;
}