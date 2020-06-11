#pragma once
#include "common.h"
namespace iotool {

struct Components {
    cv::Mat labels;
    cv::Mat centroids;
    cv::Mat x1;
    cv::Mat y1;
    cv::Mat w;
    cv::Mat h;
    cv::Mat area;
    int count;
};

class Image {
   public:
    /**
    * @param mode
    *   0:color
    *   1:gray
    *   2:unchange
   */
    Image(){};
    Image(const char* filename, int mode) {
        int modes[] = {cv::IMREAD_COLOR,
                       cv::IMREAD_GRAYSCALE,
                       cv::IMREAD_UNCHANGED};
        m_image = cv::imread(filename, modes[mode]);
    };

    Image subImage(int r1, int r2, int c1, int c2) {
        auto image = Image();
        image.m_image = m_image.clone().colRange(c1, c2).rowRange(r1, r2);
        return image;
    }
    const cv::Mat& getData() const {
        return m_image;
    }
    cv::Mat binary(float Threshold, int type = 0, int maxval = 255) {
        int types[] = {
            cv::ThresholdTypes::THRESH_BINARY,
            cv::ThresholdTypes::THRESH_BINARY_INV,
            cv::ThresholdTypes::THRESH_TOZERO,
            cv::ThresholdTypes::THRESH_TOZERO_INV};
        cv::Mat dst;
        cv::threshold(m_image, dst, Threshold, maxval, types[type]);
        return dst;
    }
    void binarize(float Threshold, int type = 0, int maxval = 255) {
        int types[] = {
            cv::ThresholdTypes::THRESH_BINARY,
            cv::ThresholdTypes::THRESH_BINARY_INV,
            cv::ThresholdTypes::THRESH_TOZERO,
            cv::ThresholdTypes::THRESH_TOZERO_INV};
        cv::Mat dst;
        cv::threshold(m_image, m_image, Threshold, maxval, types[type]);
    }
    int getWidth() const {
        return m_image.cols;
    }
    int getHeight() const {
        return m_image.rows;
    }
    Components getConnectComp(int connectivity = 8) {
        return getConnectComp(m_image, connectivity);
    };
    cv::Mat distance(int dist_type) {
        return distance(m_image, dist_type);
    };
    static Components getConnectComp(const cv::Mat& I, int connectivity = 8) {
        cv::Mat stats;
        Components cp;
        cp.count = cv::connectedComponentsWithStats(I, cp.labels, stats, cp.centroids, connectivity);
        cp.x1 = stats.colRange(0, 1);
        cp.y1 = stats.colRange(1, 2);
        cp.w = stats.colRange(2, 3);
        cp.h = stats.colRange(3, 4);
        cp.area = stats.colRange(4, 5);

        return cp;
    };
    static cv::Mat distance(cv::Mat image, int dist_type = cv::DIST_L2) {
        cv::Mat dist;
        cv::distanceTransform(image, dist, dist_type, 3);
        return dist;
    }

    cv::Mat rectangle(int x, int y, int w, int h, int thickness = 1, cv::Scalar color = cv::Scalar(0, 0, 255)) {
        auto r = m_image.clone();
        cv::Point p1(x, y), p2(x + w, y + h);
        cv::rectangle(r, p1, p2, color, thickness, cv::LINE_8);
        return r;
    }

   private:
    cv::Mat m_image;
};
}  // namespace iotool
