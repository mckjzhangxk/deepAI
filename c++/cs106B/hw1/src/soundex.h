#ifndef SOUNDEX_H
#define SOUNDEX_H
#include <iostream>
#include "vector.h"
#include "map.h"
#include "strlib.h"
using namespace std;

class Soundex
{
public:
    Soundex(string &);
    void init();
private:
    friend ostream& operator<<(ostream& out,const Soundex & v);
    Vector<char> mcodes;
    Map<char,char> CODE_DICT={
        {'A','0'},{'E','0'},{'I','0'},{'O','0'},{'U','0'},{'H','0'},{'W','0'},{'U','0'},
        {'B','1'},{'F','1'},{'P','1'},{'V','1'},
        {'C','2'},{'G','2'},{'J','2'},{'K','2'},{'Q','2'},{'S','2'},{'X','2'},{'C','Z'},
        {'D','4'},{'T','4'},
        {'M','5'},{'N','5'},
        {'L','5'},
        {'R','6'}
    };
};

#endif // SOUNDEX_H
