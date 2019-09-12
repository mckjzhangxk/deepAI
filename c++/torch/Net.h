//
// Created by zhangxk on 2019/9/12.
//

#ifndef TORCH_NET_H
#define TORCH_NET_H

#include <torch/torch.h>

struct Net :torch::nn::Module{
public:
    Net(int64_t N,int64_t M):linear(register_module("linear",torch::nn::Linear(N,M))){
        this->bias=register_parameter("b",torch::randn(M));
    }
    torch::Tensor forward(torch::Tensor x);
public:
    torch::Tensor bias;
    torch::nn::Linear linear;
};


#endif //TORCH_NET_H
