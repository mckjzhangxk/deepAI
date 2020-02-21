#include <gtest/gtest.h>
#include "image.h"

using igg::Image;
TEST(TEST_ROW_COL, ROW_COL) {
    Image img(32, 57);
    EXPECT_EQ(img.rows(), 32);
    EXPECT_EQ(img.cols(), 57);
}
TEST(TEST_AT, AT) {
    Image img(32, 57);
    img.at(3, 5) = 26;
    img.at(5, 22) = 55;
    EXPECT_EQ(26, img.at(3, 5));
    EXPECT_EQ(55, img.at(5, 22));
}
TEST(TEST_READ, READ) {
    Image img;
    bool bl = img.FillFromPgm("../data/lena.ascii_err.pgm");
    EXPECT_EQ(false, bl);

    bl = img.FillFromPgm("../data/lena.ascii.pgm");
    EXPECT_EQ(true, bl);

    EXPECT_EQ(512, img.rows());
    EXPECT_EQ(512, img.cols());
    EXPECT_EQ(162, img.at(0, 0));
    EXPECT_EQ(156, img.at(0, 5));
}
TEST(TEST_HIST, TEST_HIST_1_Test) {
    Image img;
    bool bl = img.FillFromPgm("../data/lena.ascii.pgm");
    EXPECT_EQ(true, bl);

    int bins = 7;
    std::vector<float> hist = img.ComputeHistogram(bins);
    EXPECT_EQ(bins, hist.size());
    float s = 0;
    for (float p : hist) {
        EXPECT_GE(p, 0);
        s += p;
    }
    EXPECT_NEAR(1, s, 1e-4);
}

TEST(TEST_HIST, TEST_UPSCALE) {
    Image img;
    bool bl = img.FillFromPgm("../data/lena.ascii.pgm");
    EXPECT_EQ(true, bl);

    img.UpScale(2);
    EXPECT_EQ(1024, img.rows());
    EXPECT_EQ(1024, img.cols());
}

TEST(TEST_HIST, TEST_DOWNSCALE) {
    Image img;
    bool bl = img.FillFromPgm("../data/lena.ascii.pgm");
    EXPECT_EQ(true, bl);

    img.DownScale(2);
    EXPECT_EQ(256, img.rows());
    EXPECT_EQ(256, img.cols());
}