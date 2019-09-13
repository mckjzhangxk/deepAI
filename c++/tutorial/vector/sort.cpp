#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

/*
    STL algorithm
    1.sort
    2.random_shuffle
    3.foreach
    
*/
void show(const int& x){
    cout<<x<<endl;
}
bool worseThan(const int& x,const int& y){
    return x>y;
}
int main(int argc, char const *argv[]){
    vector<int> a(10);
    for(unsigned i=0;i<a.size();i++){
        a[i]=i;
    }
    random_shuffle(a.begin(),a.begin()+5);
    for_each(a.begin(),a.end(),show);
    sort(a.begin(),a.end(),worseThan);
    for_each(a.begin(),a.end(),show);
    
    
    
}