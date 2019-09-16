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

Sequential  make_generator(int kNoiseSize){

    nn::Sequential generator(
            // Layer 1
            nn::Conv2d(nn::Conv2dOptions(kNoiseSize, 256, 4)
                               .with_bias(false)
                               .transposed(true)),
            nn::BatchNorm(256),
            nn::Functional(torch::relu),
            // Layer 2
            nn::Conv2d(nn::Conv2dOptions(256, 128, 3)
                               .stride(2)
                               .padding(1)
                               .with_bias(false)
                               .transposed(true)),
            nn::BatchNorm(128),
            nn::Functional(torch::relu),
            // Layer 3
            nn::Conv2d(nn::Conv2dOptions(128, 64, 4)
                               .stride(2)
                               .padding(1)
                               .with_bias(false)
                               .transposed(true)),
            nn::BatchNorm(64),
            nn::Functional(torch::relu),
            // Layer 4
            nn::Conv2d(nn::Conv2dOptions(64, 1, 4)
                               .stride(2)
                               .padding(1)
                               .with_bias(false)
                               .transposed(true)),
            nn::Functional(torch::tanh));
    return generator;
}

Sequential make_discriminator(){
    nn::Sequential discriminator(
            // Layer 1
            nn::Conv2d(
                    nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).with_bias(false)),
            nn::Functional(torch::leaky_relu, 0.2),
            // Layer 2
            nn::Conv2d(
                    nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).with_bias(false)),
            nn::BatchNorm(128),
            nn::Functional(torch::leaky_relu, 0.2),
            // Layer 3
            nn::Conv2d(
                    nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).with_bias(false)),
            nn::BatchNorm(256),
            nn::Functional(torch::leaky_relu, 0.2),

            // Layer 4
            nn::Conv2d(
                    nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).with_bias(false)),
            nn::Functional(torch::sigmoid));
    return discriminator;

}

int main() {
//    int M=3;
//    int N=5;
//    torch::Tensor x = torch::rand({10,M});
//
//    Net net(M,N);
//
//    OrderedDict<string,Tensor> nameparam=net.named_parameters();
//
//
//    for(auto x:net.named_parameters()){
//        cout<<x.key()<<endl;
//    }
//    Tensor y=net.forward(x);
//    cout<<y.sizes()<<endl;
    int kNumberOfEpochs=10;
    int kNoiseSize=20;



    torch::Device device = torch::kCPU;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }

    Sequential generator=make_generator(kNoiseSize);
    Sequential discriminator= make_discriminator();
    generator->to(device);
    discriminator->to(device);

    auto dataset = torch::data::datasets::MNIST("/home/zxk/AI/data/mnist/MNIST_DATA")
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(10).workers(2));
    int batches_per_epoch=55000/10;
//    cout<<dataset.size().value()<<endl;
    torch::optim::Adam generator_optimizer(
            generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
    torch::optim::Adam discriminator_optimizer(
            discriminator->parameters(), torch::optim::AdamOptions(5e-4).beta1(0.5));

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        int64_t batch_index = 0;
        for (torch::data::Example<>& batch : *data_loader) {
            // Train discriminator with real images.
            discriminator->zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0).to(device);
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();
//
//            // Train discriminator with fake images.
            torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}).to(device);
            torch::Tensor fake_images = generator->forward(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0),device);
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());
            torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();
//
            torch::Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();
//
//            // Train generator.
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();

            std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
                    epoch,
                    kNumberOfEpochs,
                    ++batch_index,
                    batches_per_epoch,
                    d_loss.item<float>(),
                    g_loss.item<float>());
        }
    }
}