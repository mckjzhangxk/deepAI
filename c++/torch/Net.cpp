//
// Created by zhangxk on 2019/9/12.
//

#include "Net.h"

torch::Tensor Net::forward(torch::Tensor x) {

    return linear(x)+bias;
}
