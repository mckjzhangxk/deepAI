#include<iostream>
#include<vector>

using namespace std;
/*
    vector
    1.begin(),end()返回的是迭代器,广义指针
    2.erase(start,end)
    3.insert(pos,new_start,new_end)
    4.swap
*/
void show(const vector<int> &a){
    cout<<"------------------------------"<<endl;
     //end指向结尾,可以理解成string 的null结尾
    for(auto pr=a.begin();pr!=a.end();pr++){
        cout<<*pr<<endl;
    }
    for(auto x:a){
        cout<<x<<endl;
    }
    cout<<"==============================="<<endl;
}
int main(int argc, char const *argv[])
{
    vector<int> a(10);
    for(unsigned i=0;i<a.size();i++){
        a[i]=i;
    }

    show(a);

    vector<int>b(a);
    a.erase(a.begin()+3,a.begin()+5);
    show(a);

    a.insert(a.begin()+3,b.begin()+3,b.begin()+5);
    
    return 0;
}
