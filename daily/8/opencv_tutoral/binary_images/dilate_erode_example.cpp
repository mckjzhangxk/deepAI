#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include "Morphological.h"
/**
 * https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html
 * 
 * I see, I'm going to try timing this against what I am currently doing which is using cv2.connectedComponents() to find the connected components, extracting the component of my point, then using cv2.findContours() and subsequently cv2.contourArea() and cv2.arcLenth(). – mv3 Jan 26 '17 at 17:23
*/
struct Setup {
    int kernel_size;
    int max_kernel_size;
    std::string dialate_window_name;
    std::string erode_window_name;
    std::string open_window_name;
    std::string close_window_name;
};
Setup setup = {
    3, 10,
    "dilate example",
    "erode example",
    "open example",
    "close example"};
Dilate dilate;
Erode erode;
Morphology morph;
cv::Mat I;

void init(const char *filename) {
    I = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (I.empty()) {
        std::cout << "can't open file" << std::endl;
        exit(0);
    }
    std::cout << "width:" << I.cols
              << ",height:" << I.rows
              << ",channels:" << I.channels()
              << std::endl;
    dilate.setKsize(setup.kernel_size);
    dilate.setType(cv::MORPH_RECT);
    erode.setKsize(setup.kernel_size);
    erode.setType(cv::MORPH_RECT);
}
void dilate_callback(int, void *);
void erode_callback(int, void *);
void open_callback(int, void *);
void close_callback(int, void *);
int main() {
    init("/home/zxk/PycharmProjects/deepAI1/daily/8/opencv_tutoral/bin/book/binary/close.png");

    cv::namedWindow(setup.dialate_window_name, cv::WINDOW_FREERATIO);
    cv::namedWindow(setup.erode_window_name, cv::WINDOW_FREERATIO);
    cv::namedWindow(setup.open_window_name, cv::WINDOW_FREERATIO);
    cv::namedWindow(setup.close_window_name, cv::WINDOW_FREERATIO);

    cv::createTrackbar(setup.dialate_window_name,
                       setup.dialate_window_name,
                       &setup.kernel_size,
                       setup.max_kernel_size,
                       dilate_callback);
    cv::createTrackbar(setup.erode_window_name,
                       setup.erode_window_name,
                       &setup.kernel_size,
                       setup.max_kernel_size,
                       erode_callback);
    cv::createTrackbar(setup.open_window_name,
                       setup.open_window_name,
                       &setup.kernel_size,
                       setup.max_kernel_size,
                       open_callback);
    cv::createTrackbar(setup.close_window_name,
                       setup.close_window_name,
                       &setup.kernel_size,
                       setup.max_kernel_size,
                       close_callback);
    dilate_callback(0, 0);
    erode_callback(0, 0);
    open_callback(0, 0);
    close_callback(0, 0);
    cv::waitKey(0);
}

void dilate_callback(int, void *) {
    if (setup.kernel_size > 0) {
        dilate.setKsize(setup.kernel_size);
        cv::Mat dst = dilate.handle(I);
        cv::imshow(setup.dialate_window_name, dst);
    }
}
void erode_callback(int, void *) {
    if (setup.kernel_size > 0) {
        erode.setKsize(setup.kernel_size);
        cv::imshow(setup.erode_window_name, erode.handle(I));
    }
}
void open_callback(int, void *) {
    if (setup.kernel_size > 0) {
        morph.setKsize(setup.kernel_size);
        cv::imshow(setup.open_window_name, morph.open(I));
    }
}
void close_callback(int, void *) {
    if (setup.kernel_size > 0) {
        morph.setKsize(setup.kernel_size);
        cv::imshow(setup.close_window_name, morph.close(I));
    }
}