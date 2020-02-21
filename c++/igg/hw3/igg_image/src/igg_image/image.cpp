#include "image.h"
using igg::Image;

Image::Image(const IoStrategy& io_strategy) : io_strategy_(io_strategy) {
}
Image::Image(int rows, int cols, const IoStrategy& io_strategy) : rows_(rows), cols_(cols), io_strategy_(io_strategy) {
    data_.reserve(rows * cols);
}

int Image::rows() const {
    return rows_;
}
int Image::cols() const {
    return cols_;
}
Image::Pixel& Image::at(int row, int col) {
    return data_[row * cols_ + col];
}
bool Image::ReadFromDisk(const std::string& file_name) {
    ImageData img = io_strategy_.Read(file_name);

    if (img.data.empty())
        return false;
    else {
        data_.clear();
        std::vector<int> reds = img.data[0];
        std::vector<int> greens = img.data[1];
        std::vector<int> blues = img.data[2];

        for (int r = 0; r < img.rows; r++)
            for (int c = 0; c < img.cols; c++) {
                int idx = r * img.cols + c;
                data_.push_back({reds[idx], greens[idx], blues[idx]});
            }

        rows_ = img.rows;
        cols_ = img.cols;
        max_val_ = img.max_val;
    }
    return true;
}

void Image::WriteToDisk(const std::string& file_name) const {
    std::vector<std::vector<int>> temp;
    std::vector<int> reds(rows_ * cols_);
    std::vector<int> greens(rows_ * cols_);
    std::vector<int> blues(rows_ * cols_);

    for (int r = 0; r < rows_; r++)
        for (int c = 0; c < cols_; c++) {
            int idx = r * cols_ + c;
            reds[idx] = data_[idx].red;
            greens[idx] = data_[idx].green;
            blues[idx] = data_[idx].blue;
        }
    temp.push_back(reds);
    temp.push_back(greens);
    temp.push_back(blues);
    ImageData d = {rows_, cols_, max_val_, temp};
    io_strategy_.Write(file_name, d);
}

// std::vector<float> Image::ComputeHistogram(int bins) const {
//     std::vector<float> hist;
//     hist.reserve(bins);
//     for (int i = 0; i < bins; i++)
//         hist.push_back(0);

//     float resolution = (1 + max_val_) / bins;
//     float score = 1. / (rows_ * cols_);
//     for (int c : data_) {
//         int index = c / resolution;
//         hist[index] += score;
//     }

//     return hist;
// }
void Image::DownScale(int scale) {
    if (scale == 1) return;

    int old_cols = cols_;
    std::vector<Image::Pixel> old_data = data_;

    init_field(rows_ / scale, cols_ / scale);

    for (int i = 0; i < rows_; i++)
        for (int j = 0; j < cols_; j++) {
            data_[i * cols_ + j] = old_data[scale * i * old_cols + scale * j];
        }
}
void Image::UpScale(int scale) {
    if (scale == 1) return;

    int old_cols = cols_;
    std::vector<Image::Pixel> old_data = data_;

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
    data_.resize(rows_ * cols_, {0, 0, 0});
}