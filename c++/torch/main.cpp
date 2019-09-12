#include <iostream>
#include <torch/torch.h>
#include "Net.h"
using  namespace std;

int main() {
//    std::cout << "Hello, World!" << std::endl;
//    torch::Tensor eye=torch::rand({1,2});
//
//    cout<<eye.sizes()<<endl;
//    cout<<eye<<endl;
    Net net(4,5);
//    for(const auto& p:net.parameters()){
//        cout<<p<<endl;
//    }
    for(const auto & p:net.named_parameters()){
        cout<<p.key()<<endl;
    }

    const torch::OrderedDict<string,torch::Tensor> dict=net.named_parameters();
    cout<<dict["b"]<<endl;
    torch::Tensor s=net.forward(torch::randn({10,4}));
//    cout<<s<<endl;
    return 0;
}