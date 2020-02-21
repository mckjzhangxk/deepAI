add_library(strategies
            # TODO: Add other files here when needed.
            ppm_iostrategy.cpp
            png_strategy.cpp
            dummy_strategy.cpp)
# TODO: link strategies against needed libraries.
target_link_libraries(strategies png)