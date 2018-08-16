#include <iostream>
#include <complex>
using namespace std;

template <class T>

T sum(T a[],int n){
	T r=0;
	for(int i=0;i<n;i++)
		r+=a[i];
	return r;
}

int main()
{
	int a[4]={1,2,3,4};
	int sa=sum(a,4);
	cout<<"sum1 is"<<sa<<endl;	

	double b[4]={2.0,2.5,3.0,4.0};
	double sb=sum(b,4);
	cout<<"sum2 is"<<sb<<endl;
	//for complex number
	complex<double> c[4]={};
	for(int i=0;i<4;i++){
		c[i]=complex<double>(i+1,i+1);	
		cout<<"complex"<<i<<" is"<<c[i]<<endl;

	}
	complex<double> cb=sum(c,4);	
	cout<<"sum3 is"<<cb<<endl;	
}

