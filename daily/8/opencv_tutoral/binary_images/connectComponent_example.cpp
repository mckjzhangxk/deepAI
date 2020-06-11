#include <cstdlib>

#include "Image.h"
#include "gui.h"

static void callback(int value, void *userdata) {
    AppProp *p = (AppProp *)userdata;
    iotool::Image &im = *(iotool::Image *)p->userdata;
    cv::Mat bim = im.binary(value, 1);
    iotool::Components st = iotool::Image::getConnectComp(bim);

    cv::Mat_<cv::Vec3b> I(im.getHeight(), im.getWidth());
    std::vector<cv::Vec3b> colors;

    colors.push_back({0, 0, 0});
    for (int i = 1; i < st.count; i++) {
        colors.push_back({rand() % 255, rand() % 255, rand() % 255});
    }

    for (int i = 0; i < im.getHeight(); i++)
        for (int j = 0; j < im.getWidth(); j++) {
            int color = st.labels.at<int>(i, j);
            I.at<cv::Vec3b>(i, j) = colors[color];
        }
    cv::cvtColor(I, I, cv::COLOR_RGB2BGR);
    cv::imshow(p->window_name, I);
}
int main() {
    char *filename = "/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/bin/book/binary/ocr2.png";
    iotool::Image I(filename, 1);
    AppProp p = {"", 128, 255, &I};

    SliderWindow(p, callback);
    cv::waitKey();
}