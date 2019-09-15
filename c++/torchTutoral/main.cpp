#include <iostream>
#include <torch/torch.h>
#include <cublas_v2.h>

using namespace torch::nn;
using namespace torch;
using namespace std;

struct  Net:public torch::nn::Module{
    Net(unsigned M, unsigned N)
        :linear(register_module("linear",Linear(M,N)))
    {
        another_bias=register_parameter("bias",torch::rand(N));
//        linear=register_module("linear",Linear(M,N));
    }
    torch::Tensor forward(torch::Tensor x){
        return linear(x)+another_bias;
    }
    torch::nn::Linear linear;
    torch::Tensor another_bias;
};
int main() {
    int M=3;
    int N=5;
    torch::Tensor x = torch::rand({10,M});

    Net net(M,N);

    OrderedDict<string,Tensor> nameparam=net.named_parameters();
//    for(OrderedDict<string,Tensor>::ConstIterator  bp=nameparam.begin();bp!=nameparam.end();bp++){
//        cout<<nameparam.<<endl;
//    }
    for(auto x:net.named_parameters()){
        cout<<x.key()<<endl;
    }
    Tensor y=net.forward(x);
    cout<<y.sizes()<<endl;
}