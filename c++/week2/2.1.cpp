#include <iostream>
using namespace std;
/*
	Three topic in this course2.1
	1.default argument value
	2.const modifier
	3.more generic type

*/
template<class sumable>
sumable sum(const sumable a[],int size,sumable s0=0){
	sumable r=s0;
	for (int i=0;i<size;i++){
		r+=a[i];	
	}	
	return r;
}

int main(int argc,char **argv){
	//play with it,see error when assign a[0] the value	
	//const int a[]={1,2,3};
	//a[0]=1;
	int a[]={1,2,3};
	double d[]={1,2,3};
	cout<<"sum of integer is "<<sum(a,3)<<endl;
	cout<<"sum of double is "<<sum(d,3)<<endl;
	cout<<"sum of double with initial value 20 is "<<sum(d,3,20.0)<<endl;//20 will accure a error of conflict
	return 0;
}

