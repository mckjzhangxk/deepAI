#include<iostream>
#include<string>
#include<map>
#include<set>
#include<algorithm>

using namespace std;

int main(int argc, char const *argv[]){
    map<string,string> m;
    m.insert(pair<string,string>("name","zxk"));
    m.insert(pair<string,string>("age","12"));

    for(map<string,string>::iterator it=m.begin();it!=m.end();it++){
        cout<<(*it).first<<":"<<(*it).second<<endl;
    }
    string key="name";
    if(m.find(key)!=m.end()){
        cout<<"found"<<endl;
        m[key]="yes";
        cout<<m[key]<<endl;
    }


}