#ifndef MYUTILS_H
#define MYUTILS_H
#include "commonHeader.h"


bool checkError(int id);
bool api_init();
/*
   create vertex shader and fragment shader
   vsfile:source code of vertex shader
   fsfile:source code of fragment shader

   return:
    a program
*/
int createShader(std::string& vsfile,std::string& fsfile);
#endif // MYUTILS_H
