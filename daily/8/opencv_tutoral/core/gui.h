#pragma once
#include "common.h"
struct AppProp {
    char *window_name;
    int t;
    int maxval;
    void *userdata;
};
#define SliderWindow(p, cb)                                                   \
    cv::namedWindow(p.window_name, cv::WINDOW_FREERATIO);                     \
    cv::createTrackbar(p.window_name, p.window_name, &p.t, p.maxval, cb, &p); \
    callback(p.t, &p);