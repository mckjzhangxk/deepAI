#include <iostream>
using namespace std;
#ifndef BAD
#define BAD

class BadString{
public:
BadString(const char *);
~BadString();
BadString(const BadString &);
static int count;

friend ostream & operator<<(ostream &out,const BadString &s);
const BadString& operator=(const BadString &);
private:
char *m_char;
};

#endif
