#include<iostream>
#include<string>
#include<unordered_map>
#include<unordered_set>
#include<algorithm>
using namespace std;

int main(int argc, char const *argv[])
{
    unordered_map<string,string>m ={{"name","zxk"},{"address","jinan"}};
    //这是const iter和非 const的区别 
    unordered_map<string,string> ::iterator it= m.find("parent");
    (*it).second="xxx";

    if(it==m.end()){
        m["parent"]="yzy";
    }

    for(it=m.begin();it!=m.end();it++){
        cout<<(*it).first<<":"<<(*it).second<<endl;;
    }
    return 0;
}
