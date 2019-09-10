#include<iostream>
#include<sstream>
#include<GL/glut.h>
#include<memory.h>
#include<string>
#include<vecmath/Matrix4f.h>


GLfloat angle=0;
GLfloat pangle=0;
GLfloat aspect=1.7;
int INTERVAL=100;
bool rotate=0;

void timefunc(int value){
    if(rotate){
        angle+=0.01;
    
    }
    glutPostRedisplay();
    
    glutTimerFunc(INTERVAL,timefunc,value);
}

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

    glMatrixMode(GL_MODELVIEW);
    // Matrix4f rot=rot.rotateZ(angle);
     

    glLoadIdentity();
    // glMultMatrixf(rot);
    // glTranslatef(0.5,0.5,0);


    gluLookAt(0,0,5,0,0,0,0,1,0);
    glBegin(GL_TRIANGLES);
        glColor3f(0,1,0);
        glVertex3d(0,0,0);
        glNormal3d(0,0,1);

        glVertex3d(1,0,0);
        glNormal3d(0,0,1);

        glVertex3d(1,1,0);
        glNormal3d(0,0,1);

    glEnd();


    glBegin(GL_TRIANGLES);
        glColor3f(0,1,1);
        glVertex3d(0,0,0);
        glNormal3d(0,0,1);

        glVertex3d(-1,0,0);
        glNormal3d(0,0,1);

        glVertex3d(-1,-1,0);
        glNormal3d(0,0,1);
            

    glEnd();

    
    glBegin(GL_TRIANGLES);
        glColor3f(1,1,1);
        glVertex3d(0,2,0);
        glNormal3d(0,0,1);

        glVertex3d(-1,-1,0);
        glNormal3d(0,0,1);

        glVertex3d(1,-1,0);
        glNormal3d(0,0,1);
    glEnd();

    glutSwapBuffers();
}
void reshape(int w,int h){
  
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
      
    glViewport(0,0,w,h);
    cout<<aspect<<endl;
    gluPerspective(pangle, aspect, 1.0, 100.0);
   
}
void keyFunc(unsigned char key,int x, int y){
    switch (key)
    {
    case 'q':
        exit(0);
        break;
    case 'r':
        rotate=1-rotate;
        cout<<rotate<<endl;
        break;
    case 'x':
        pangle+=1;
        reshape(glutGet(GLUT_WINDOW_WIDTH),glutGet(GLUT_WINDOW_HEIGHT));
        break;
    case 'z':
        aspect*=0.9;
        reshape(glutGet(GLUT_WINDOW_WIDTH),glutGet(GLUT_WINDOW_HEIGHT));
        break;
    default:
        break;
    }
    glutPostRedisplay();
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
    glEnable(GL_DEPTH_TEST);   // Depth testing must be turned on
    
    glutCreateWindow("Hello Computer Graphics");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyFunc);
    glutMouseFunc(mouseFunc);
    //glutTimerFunc(INTERVAL,timefunc,1);
    glutMainLoop();
    return 0;
}
