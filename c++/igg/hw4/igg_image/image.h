// Copyright Igor Bogoslavskyi, year 2018.
// In case of any problems with the code please contact me.
// Email: igor.bogoslavskyi@uni-bonn.de.

#pragma once

#include <memory>
#include <vector>
#include "strategy.h"
namespace igg {

class Image {
   public:
    /// A struct within class Image that defines a pixel.
    struct Pixel {
        int red;
        int green;
        int blue;
    };

    // TODO: fill public interface.
    Image(const std::shared_ptr<IoStrategy>& io_strategy);
    Image(int rows, int cols, const std::shared_ptr<IoStrategy>& io_strategy);
    int rows() const;
    int cols() const;
    int maxValue() const { return max_val_; };
    Pixel& at(int row, int col);

    void DownScale(int scale);
    void UpScale(int scale);

    bool ReadFromDisk(const std::string& file_name);
    void WriteToDisk(const std::string& file_name) const;
    void set_io_strategy(const std::shared_ptr<IoStrategy>& io);

   private:
    // TODO: add missing private members when needed.
    void init_field(int rows, int cols);
    int rows_ = 0;
    int cols_ = 0;
    int max_val_ = 255;
    std::vector<Pixel> data_;
    std::shared_ptr<IoStrategy> io_strategy_;
};

}  // namespace igg
