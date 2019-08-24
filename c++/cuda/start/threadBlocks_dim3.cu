#include <iostream>
#include <math.h>
#include <cuda.h>
#include <stdio.h>
// function to add the elements of two arrays
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}
__global__ void add(int n, float *x, float *y)
{

}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];
  cudaMallocManaged(&x,N*sizeof(float));
  cudaMallocManaged(&y,N*sizeof(float));


  // Run kernel on 1M elements on the CPU
  const dim3 threadsPerBlock(16, 16);
  const dim3 numBlocks(1024/16, 768/16);
  add<<<numBlocks,threadsPerBlock>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());  
    // Free memory
  cudaFree(x);
  cudaFree(y);
  // Free memory
  return 0;
}
