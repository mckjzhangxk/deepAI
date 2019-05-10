#include "BadString.h"
#include <cstring>
#include <iostream>


using namespace std;

int BadString::count=0;


BadString::BadString(const char * a){
  int n=strlen(a)+1;
  m_char=new char[n];
  strcpy(m_char,a);
  count+=1;
  cout<<"create:"<<count<<endl;
}
BadString::BadString(const BadString &a){
  int n=strlen(a.m_char)+1;
  m_char=new char[n];
  strcpy(m_char,a.m_char);
  count+=1;
  cout<<"create:"<<count<<endl;
}
BadString::~BadString(){
  delete m_char;
  count-=1;
  cout<<"delete:"<<count<<endl;
}
ostream & operator<<(ostream &out,const BadString &s){
 out<<s.m_char; 
 return out;
}

const BadString& BadString::operator=(const BadString & s){
  if (this==&s) return s;
  
  delete m_char;
  int n=strlen(s.m_char)+1;
  m_char=new char[n];
  strcpy(m_char,s.m_char);
  return *this;
}
void f(BadString  s){
}
int main(){
 BadString s1("Hello");
 BadString s2("World");
 BadString s3{"NB"};
 f(s3);
 BadString s4=s3;
 cout<<s1<<endl;
 cout<<s2<<endl; 
 cout<<s3<<endl;
 cout<<s4<<endl;
 s1=s2;
 cout<<s1<<endl;
}
