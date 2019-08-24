#include<string>
#include<iostream>

using namespace std;
int main(){

 string a="123/456/789";
 string delimiter="/";
 int start=8;
 int pos=a.find(delimiter,pos=start);
 cout<<pos<<endl;
 cout<<a.substr(start,pos-start)<<endl;
}
