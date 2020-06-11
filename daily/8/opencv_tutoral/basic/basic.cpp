#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

/**
 * 演示怎么读取一个文件
*/
void readDemo(const char* name) {
    cv::Mat I = cv::imread(name, cv::IMREAD_COLOR);
    if (I.empty()) {
        std::cout << "can't open file" << std::endl;
        exit(0);
    }
    cv::imshow("demo1", I);
    cv::waitKey();

    cv::Mat ZI = cv::Mat::zeros(I.rows, I.cols, I.type());
    cv::imshow("demo1", ZI);
    cv::waitKey();
}
/**
 * 
 * 演示写文件,转换图片
 * 以及矩阵属性的操作
*/
void writeDemo(const char* name_in, const char* name_out) {
    cv::Mat I = cv::imread(name_in, cv::IMREAD_COLOR);
    cv::Mat dst(I.rows, I.cols, I.type());

    cv::cvtColor(I, dst, cv::COLOR_BGR2GRAY);
    cv::imwrite(name_out, dst);
}
int main(int argc, char* argv[]) {
    if (argc > 1) {
        readDemo(argv[1]);
    }
    if (argc > 2) {
        writeDemo(argv[1], argv[2]);
    }
}