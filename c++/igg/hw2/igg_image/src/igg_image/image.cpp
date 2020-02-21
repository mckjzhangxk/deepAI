#include "image.h"
#include "io_tools.h"
using igg::Image;
using igg::io_tools::ImageData;
Image::Image() {
}
Image::Image(int rows, int cols) : rows_(rows), cols_(cols) {
    data_.reserve(rows * cols);
}
int Image::rows() const {
    return rows_;
}
int Image::cols() const {
    return cols_;
}
int& Image::at(int row, int col) {
    return data_[row * cols_ + col];
}
bool Image::FillFromPgm(const std::string& file_name) {
    ImageData img = igg::io_tools::ReadFromPgm(file_name);

    if (img.data.empty())
        return false;
    else {
        data_ = img.data;
        rows_ = img.rows;
        cols_ = img.cols;
        max_val_ = img.max_val;
    }
    return true;
}

void Image::WriteToPgm(const std::string& file_name) const {
    ImageData d = {rows_, cols_, max_val_, data_};
    igg::io_tools::WriteToPgm(d, file_name);
}

std::vector<float> Image::ComputeHistogram(int bins) const {
    std::vector<float> hist;
    hist.reserve(bins);
    for (int i = 0; i < bins; i++)
        hist.push_back(0);

    float resolution = (1 + max_val_) / bins;
    float score = 1. / (rows_ * cols_);
    for (int c : data_) {
        int index = c / resolution;
        hist[index] += score;
    }

    return hist;
}
void Image::DownScale(int scale) {
    if (scale == 1) return;

    int old_cols = cols_;
    std::vector<int> old_data = data_;

    init_field(rows_ / scale, cols_ / scale);

    for (int i = 0; i < rows_; i++)
        for (int j = 0; j < cols_; j++) {
            data_[i * cols_ + j] = old_data[scale * i * old_cols + scale * j];
        }
}
void Image::UpScale(int scale) {
    if (scale == 1) return;

    int old_cols = cols_;
    std::vector<int> old_data = data_;

    init_field(scale * rows_, scale * cols_);
    for (int i = 0; i < rows_; i++)
        for (int j = 0; j < cols_; j++) {
            data_[i * cols_ + j] = old_data[i / scale * old_cols + j / scale];
        }
}
void Image::init_field(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    data_.clear();
    data_.reserve(rows_ * cols_);
    data_.resize(rows_ * cols_, 0);
}