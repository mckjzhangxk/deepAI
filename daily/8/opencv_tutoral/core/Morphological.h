#pragma once
#include "common.h"

class MorphBase {
   public:
    /**
     * @size: kernel的大小
     * @type: kernel的形状
     * Dilate表示一个扩张操作,dst(x,y)=max(src(x+dx,y+dy)),dx,dy不是0,前景色吞并
     * 背景色
    */
    MorphBase(int size = 3, int type = cv::MORPH_RECT) : m_kernel_size(size), m_type(type) {
    }
    virtual ~MorphBase(){};
    void setType(int type) {
        m_type = type;
    }

    void setKsize(int size) {
        m_kernel_size = size;
    }
    cv::Point getAnchor() const {
        return cv::Point(m_kernel_size / 2, m_kernel_size / 2);
    };
    cv::Size getSize() const {
        return cv::Size(m_kernel_size, m_kernel_size);
    };
    cv::Mat getKernel() const {
        return cv::getStructuringElement(m_type, getSize(), getAnchor());
    }
    virtual cv::Mat handle(const cv::Mat&){};

   protected:
    int m_type;
    int m_kernel_size;
};
class Dilate : public MorphBase {
   public:
    /**
     * @size
     * @type
     * Dilate表示一个扩张操作,dst(x,y)=max(src(x+dx,y+dy)),dx,dy不是0,前景色吞并
     * 背景色
    */

    virtual cv::Mat handle(const cv::Mat& src) override {
        cv::Mat dst;
        cv::dilate(src, dst, getKernel());
        return dst;
    }
};
class Erode : public MorphBase {
   public:
    /**
     * @size
     * @type
     * Erode表示一个背景腐蚀前景的操作,dst(x,y)=min(src(x+dx,y+dy)),dx,dy不是0,背景色吞并
     * 前景
    */

    virtual cv::Mat handle(const cv::Mat& src) override {
        cv::Mat dst;
        cv::erode(src, dst, getKernel());
        return dst;
    }
};

class Morphology : public MorphBase {
   public:
    /**
    * open操作是erode+dilate的组合
    * erode会把离散的前景变成背景,与此同时,前景的轮廓被吞噬.
    * 执行dilate会还原前景的轮廓.
   */
    cv::Mat open(const cv::Mat& src) {
        cv::Mat dst;
        cv::morphologyEx(src, dst, cv::MORPH_OPEN, getKernel());
        return dst;
    };
    /**
    * close操作是dilate+erode的组合
    * dilate会把填充混合在前景的"小洞",与此同时,前景的轮廓扩展.
    * 执行erode会还原前景的轮廓.
   */
    cv::Mat close(const cv::Mat& src) {
        cv::Mat dst;
        cv::morphologyEx(src, dst, cv::MORPH_CLOSE, getKernel());
        return dst;
    };
};