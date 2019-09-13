#include<iostream>
#include<string>
using namespace std;
int main(int argc, char const *argv[])
{
    string s1="Good bye";

    char x=s1.front();
    s1[2]='x';
    s1.at(2)='Y';


    //assign
    string s2;

    s2.assign(s1);//Good bye
    cout<<s2<<endl;
    s2.assign(s1,5,3);//bye
    cout<<s2<<endl;
    s2.assign("Goood bye");//Good bye
    cout<<s2<<endl;
    s2.assign("Good bye",5,3);//bye
    cout<<s2<<endl;

    s2="good bye";
    cout<<s2.substr(3,4)<<endl;//d by
    s2.replace(s2.begin(),s2.begin()+4,"bye");
    cout<<s2<<endl;
}