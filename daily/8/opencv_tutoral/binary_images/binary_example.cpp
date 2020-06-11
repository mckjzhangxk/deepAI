#include <cstdlib>
#include <functional>
#include <string>

#include "Image.h"
#include "gui.h"

void callback(int value, void *userdata) {
    AppProp *p = (AppProp *)userdata;
    iotool::Image *I = (iotool::Image *)p->userdata;
    cv::imshow(p->window_name, I->binary(value, 1, p->maxval));
}
int main() {
    char *filename = "/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/bin/book/binary/ocr2.png";
    auto I = iotool::Image(filename, 1);

    AppProp p = {filename, 128, 255, &I};

    SliderWindow(p, callback);
    cv::waitKey(0);
}