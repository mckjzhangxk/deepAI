#include <gtest/gtest.h>
#include "image.h"
#include "png_strategy.h"
#include "ppm_iostrategy.h"

using igg::Image;
using igg::PngIoStrategy;
using igg::PpmIoStrategy;
TEST(TEST_READ, READ) {
    PpmIoStrategy ppms;
    Image img(32, 57, ppms);
    EXPECT_EQ(255, img.maxValue());
    PngIoStrategy pngs;
    Image img1(32, 57, pngs);
    EXPECT_EQ(255, img.maxValue());
}
TEST(TEST_WRITE, WRITE) {
    int w = 1024, h = 768;
    PpmIoStrategy ppms;
    Image img(h, w, ppms);
    EXPECT_EQ(255, img.maxValue());
    img.WriteToDisk("../data/test.ppm");
    PngIoStrategy pngs;
    w = 1024;
    h = 768;
    Image img1(h, w, pngs);
    img1.WriteToDisk("../data/test.png");
}

TEST(TEST_ROW_COL, ROW_COL) {
    PpmIoStrategy ppms;
    Image img(32, 57, ppms);
    EXPECT_EQ(img.rows(), 32);
    EXPECT_EQ(img.cols(), 57);
}
TEST(TEST_AT, AT) {
    PpmIoStrategy ppms;
    Image img(32, 57, ppms);
    img.at(3, 5) = {23, 32, 23};
    img.at(5, 22) = {45, 88, 61};
    EXPECT_EQ(23, img.at(3, 5).red);
    EXPECT_EQ(88, img.at(5, 22).green);
}

TEST(TEST_HIST, TEST_UPSCALE) {
    PpmIoStrategy ppms;
    Image img(ppms);
    bool bl = img.ReadFromDisk("../data/pbmlib.ascii.ppm");
    EXPECT_EQ(true, bl);
    EXPECT_EQ(255, img.maxValue());
    img.UpScale(2);
    EXPECT_EQ(255, img.maxValue());
    img.WriteToDisk("../data/upscale.ppm");
    EXPECT_EQ(600, img.rows());
    EXPECT_EQ(600, img.cols());
}

TEST(TEST_HIST, TEST_DOWNSCALE) {
    PpmIoStrategy ppms;
    Image img(ppms);
    bool bl = img.ReadFromDisk("../data/pbmlib.ascii.ppm");
    EXPECT_EQ(255, img.maxValue());
    EXPECT_EQ(true, bl);

    img.DownScale(2);
    EXPECT_EQ(255, img.maxValue());
    img.WriteToDisk("../data/downscale.ppm");
    EXPECT_EQ(150, img.rows());
    EXPECT_EQ(150, img.cols());
}

TEST(TEST_HIST, TEST_UPSCALE1) {
    PngIoStrategy ppms;
    Image img(ppms);
    bool bl = img.ReadFromDisk("../data/3.png");
    EXPECT_EQ(true, bl);

    img.UpScale(2);
    EXPECT_EQ(255, img.maxValue());
    img.WriteToDisk("../data/upscale.png");
    EXPECT_EQ(2 * 667, img.rows());
    EXPECT_EQ(1000 * 2, img.cols());
}

TEST(TEST_HIST, TEST_DOWNSCALE1) {
    PngIoStrategy ppms;
    Image img(ppms);
    bool bl = img.ReadFromDisk("../data/3.png");

    EXPECT_EQ(true, bl);

    img.DownScale(2);
    EXPECT_EQ(255, img.maxValue());
    img.WriteToDisk("../data/downscale.png");
    EXPECT_EQ(667 / 2, img.rows());
    EXPECT_EQ(1000 / 2, img.cols());
}
