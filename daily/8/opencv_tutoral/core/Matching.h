#pragma once
#include "Image.h"
#include "common.h"
struct MatchingResult {
    int x;
    int y;
    int w;
    int h;
    float value;
};

class Matching {
   public:
    virtual MatchingResult search(const iotool::Image& src, const iotool::Image& temp){};
};

class TempMatch : Matching {
   public:
    virtual MatchingResult search(const iotool::Image& src, const iotool::Image& temp) override {
        int r = src.getHeight() - temp.getHeight() + 1;
        int c = src.getWidth() - temp.getWidth() + 1;
        cv::Mat result;
        result.create(r, c, CV_32FC1);

        cv::matchTemplate(src.getData(), temp.getData(), result, cv::TM_CCOEFF_NORMED);
        double maxv;
        cv::Point max_loc;
        cv::minMaxLoc(result, 0, &maxv, 0, &max_loc);

        MatchingResult ret;
        ret.x = max_loc.x;
        ret.y = max_loc.y;
        ret.w = temp.getWidth();
        ret.h = temp.getHeight();
        ret.value = maxv;
        return ret;
    };
};