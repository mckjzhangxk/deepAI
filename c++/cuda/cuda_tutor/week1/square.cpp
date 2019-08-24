#include<iostream>

using namespace std;


void square(unsigned int* a,int n){
for(int i=0;i<n;i++){
a[i]=a[i]*a[i];
}
}
int main(int argc,char ** argv){

  int N=64;
  unsigned int A[N];

  for(int i=0;i<N;i++){
    A[i]=i;
  }

  square(A,N);
  for(int i=0;i<N;i++){
   cout<<i<<":"<<A[i]<<endl;
  }
}


