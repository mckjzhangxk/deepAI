#include <iostream>
using namespace std;
/*
	Two topic in this course2.2
	1.multi -generic type
	2.four type of cast
	(1)static_cast<T>()
	(2)dynamic_cast<T>()

	//unsafe cast
	(3)const_cast
	(4)reinterpret_cast
*/
template<class T1,class T2>

void copy(const T1 *source,T2 *target,int size){
	for (int i=0;i<size;i++){
		target[i]=static_cast<T2>(source[i]);
	}
}
int main(int argc,char **argv){
	int desc[3]={};
	double src[3]={2.5,3.0,4.0};
	copy(src,desc,3);
	cout<<"after copy dec:"<<endl;
	for(int i=0;i<3;i++){
		cout<<desc[i];
	}
	cout<<endl;
	return 0;
}

