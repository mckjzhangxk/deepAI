#include<iostream>
#include<string.h>

using namespace std;
//https://www.youtube.com/watch?v=hq8Io93GDOg
class MyString

{
private:
    char *mdata;
    int msize;
public:
    MyString(char* s){
        msize=strlen(s);
        mdata=new char[msize+1];
        strcpy(mdata,s);
        
    };
    MyString(const MyString& rhs):msize(rhs.msize){//copy constructor
        mdata=new char[rhs.msize+1];
        strcpy(mdata,rhs.mdata);
        cout<<"copy constructor"<<endl;
    }
    MyString(MyString&& rhs):msize(rhs.msize){//move constructor
        mdata=new char[rhs.msize+1];
        mdata=rhs.mdata;
        rhs.mdata=0;
        cout<<"xxxxxxxxxxxxxxxxxxxx"<<endl;
      
        cout<<"move constructor"<<endl;
    }
    const MyString& operator=(const MyString& rhs){
 
        if(&rhs==this){
            return *this;
        }
        delete mdata;
        msize=rhs.msize;
        mdata=new char[msize];
        strcpy(mdata,rhs.mdata);
    }
    ~MyString(){
        delete mdata;
    };
    friend std::ostream& operator<<(std::ostream & out,const MyString& s){
        out<<s.mdata;
        return out;
    }
};


MyString create(){
    MyString d("create abc");
    return d;
}
void foo(MyString f){
    cout<<"foo:"<<f<<endl;
}
int main(int argc, char const *argv[])
{
    MyString s=create();
    
    foo(s);


    foo(std::move(s));
    
    
    cin.get();
    return 0;
}
