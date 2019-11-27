#include "config.h"

#ifdef USE_GLEW
    #include<GL/glew.h>
#else
    #include <GL/gl3w.h>
    #include "gl3w.c"
#endif



#ifdef USE_GLUT
    #include <GL/glut.h>
#else
    #include <GLFW/glfw3.h>
#endif

#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
#ifdef USE_GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(640, 480);
    glutCreateWindow("cookie");
#else
    if(!glfwInit()){
        cerr<<"fail to init window system"<<endl;
    }
    GLFWwindow* window;
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    glfwMakeContextCurrent(window);
#endif


#ifdef USE_GLEW
     if (glewInit()) {
           fprintf(stderr, "failed to initialize glew\n");
           return -1;
     }
#else
    if(gl3wInit()){
        fprintf(stderr, "failed to initialize gl3w\n");
        return -1;
    }
#endif

     const unsigned char * vender=glGetString(GL_VENDOR);
     const unsigned char *version=glGetString(GL_VERSION);

     cout<<"vender:"<<vender<<endl;
     cout<<"opengl Version:"<<version<<endl;

}
