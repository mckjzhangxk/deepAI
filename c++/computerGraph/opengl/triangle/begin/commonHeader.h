#ifndef COMMONHEADER_H
#define COMMONHEADER_H
//#define USE_GLEW
#ifdef USE_GLEW
    #include<GL/glew.h>
#else
    #include <GL/gl3w.h>
    #ifndef USE_GL3W
        #define USE_GL3W
    #endif
#endif
#include <GLFW/glfw3.h>
#include <iostream>

#endif // COMMONHEADER_H
