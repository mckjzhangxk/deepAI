#include<iostream>
#include<sstream>
#include<GL/glut.h>
#include<memory.h>
#include<string>

using namespace std;
//https://www.opengl.org/resources/libraries/glut/spec3/spec3.html
//https://www.khronos.org/registry/OpenGL-Refpages/gl4/
void parse(int argc,char **argv){
    int top=30,left=30,width=1024,height=600;

    
    // 第一步窗口的初始化等
    glutInitWindowPosition(left,top);
    glutInitWindowSize(width,height);
}

void display(){
    glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
    cout<<"display"<<endl;
    
    glBegin(GL_TRIANGLES);
        glVertex3d(0,0,0);
        glNormal3d(0,0,1);

        glVertex3d(1,0,0);
        glNormal3d(0,0,1);

        glVertex3d(1,1,0);
        glNormal3d(0,0,1);

    glEnd();


    glBegin(GL_TRIANGLES);
        glVertex3d(0,0,0);
        glNormal3d(0,0,1);

        glVertex3d(-1,0,0);
        glNormal3d(0,0,1);

        glVertex3d(-1,-1,0);
        glNormal3d(0,0,1);

    glEnd();

    glutSwapBuffers();
}
void reshape(int w,int h){
    cout<<"reshape:("<<w<<","<<h<<")"<<endl;
}
void keyFunc(unsigned char key,int x, int y){
    switch (key)
    {
    case 'q':
        exit(0);
        break;
    
    default:
        break;
    }
}
void mouseFunc(int button, int state,int x, int y){
    /*
        button:左，中，右
        state:down,up
    */
    cout<<(button==GLUT_LEFT_BUTTON)<<endl;
    cout<<"mouseFunc:("<<x<<","<<y<<")"<<endl;
    cout<<(state==GLUT_DOWN)<<endl;GLUT_DOWN;
}
int main(int argc,char **argv){
    parse(argc,argv);

    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
    glutInit(&argc,argv);

    glutCreateWindow("Hello Computer Graphics");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyFunc);
    glutMouseFunc(mouseFunc);
    glutMainLoop();
    return 0;
}
