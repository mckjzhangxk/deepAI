#include <memory>
#include "image.h"
#include "io_strategies/png_strategy.h"
#include "io_strategies/ppm_iostrategy.h"
#include "io_strategies/strategy.h"
int main(int argc, char* argv[]) {
    using igg::Image;
    using igg::IoStrategy;
    using igg::PngIoStrategy;
    using igg::PpmIoStrategy;
    using std::shared_ptr;

    shared_ptr<IoStrategy> strategy = std::make_shared<PngIoStrategy>();
    Image img(strategy);
    img.ReadFromDisk("../data/3.png");
    img.WriteToDisk("../data/3_png.png");

    shared_ptr<IoStrategy> strategy_ppm = std::make_shared<PpmIoStrategy>();
    img.set_io_strategy(strategy_ppm);
    img.WriteToDisk("../data/3_ppm.ppm");
}