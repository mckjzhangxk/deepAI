
#include<iostream>
#include<string>
using namespace std;

void foo(int x){cout << "foo_int" << endl;}
void foo(char *x){cout << "foo_char*" << endl;}

int main(int argc, char const *argv[])
{
    foo(0);
    foo((char*)0);
    foo(nullptr);
    return 0;
}
