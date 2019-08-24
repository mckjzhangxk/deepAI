#include<iostream>
#include<cuda.h>

using namespace std;

__global__
void square(unsigned int* a,int n){
	int i=threadIdx.x;
	a[i]=a[i]*a[i];
}

int main(int argc,char ** argv){

  int N=1<<20;
  unsigned int A[N];

  //cpu initialization
  for(int i=0;i<N;i++){
    A[i]=i;
  }

  int arraysize=N*sizeof(unsigned int);
  
  //cuda memory
  unsigned int * d_in;
  cudaMalloc(&d_in,arraysize);
  //copy data
  cudaMemcpy(d_in,A,arraysize,cudaMemcpyHostToDevice);

  //lanch kernel
  square<<<1,N>>>(d_in,N);

  //copy back
  cudaMemcpy(A,d_in,arraysize,cudaMemcpyDeviceToHost);
  /*for(int i=0;i<N;i++){
   cout<<i<<":"<<A[i]<<endl;
  }*/
  int a;
  cin>>a;
  cudaFree(d_in);
}


