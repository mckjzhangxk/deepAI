#include <cstdlib>

#include "Image.h"
#include "gui.h"
#include "iostream"
static void callback(int value, void *userdata) {
    AppProp *p = (AppProp *)userdata;
    iotool::Image *image = (iotool::Image *)p->userdata;

    auto I = image->binary(value, cv::THRESH_BINARY_INV);
    I = iotool::Image::distance(I, cv::DIST_C);

    std::cout << I << std::endl;
    cv::normalize(I, I, 0, 1.0, cv::NORM_MINMAX);
    cv::imshow(p->window_name, I);
}
int main() {
    char *filename = "/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/bin/book/binary/ocr1.png";
    iotool::Image I(filename, 1);
    AppProp p = {"", 128, 255, &I};

    SliderWindow(p, callback);
    cv::waitKey();
}