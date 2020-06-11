#include <cstdlib>

#include "Image.h"
#include "Matching.h"
#include "gui.h"
struct MyData {
    iotool::Image *I;
    iotool::Image *temp;
};
static void callback(int value, void *userdata) {
    AppProp *p = (AppProp *)userdata;
    MyData *d = static_cast<MyData *>(p->userdata);

    TempMatch match;
    auto result = match.search(*d->I, *d->temp);
    std::cout << result.x << "," << result.y << ",v:" << result.value << std::endl;
    std::cout << result.w << "," << result.h << ",v:" << result.value << std::endl;
    cv::imshow(p->window_name, d->I->rectangle(result.x, result.y, result.w, result.h, 2));
}

int main() {
    char *filename = "/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/bin/book/search/car.png";
    char *t_filename = "/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/bin/book/search/car1.png";
    iotool::Image I(filename, 1);

    iotool::Image temp(t_filename, 1);
    MyData mydata = {&I, &temp};
    AppProp p = {"", 128, 255, &mydata};

    SliderWindow(p, callback);
    cv::waitKey();
}