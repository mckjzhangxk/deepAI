#include <iostream>
#include <vector>
using namespace std;

/*
	main topic is STL:
	standard template Library

	vector is a sequence container
*/

int main(int argc,char * argv[]){
	//http://www.cplusplus.com/reference/vector/vector/
	vector<double> v(10);	
	//v.push_back(1);
	
	for(int i=0;i<v.size();i++){
		v[i]=i+1;
		cout<<v[i]<<endl;
	}
		
	for(vector<double>::iterator it=v.begin();it!=v.end();it++){
		cout<<*it<<endl;
	}

}
