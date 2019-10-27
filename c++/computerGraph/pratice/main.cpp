#include<iostream>
#include<sstream>
#include<GL/glut.h>
#include<memory.h>
#include<string>
#include<vecmath/Matrix4f.h>
#include<vecmath/Vector3f.h>
#include<vector>
#include "draw.h"
#include "parse.h"

#include "Camera.h"

using namespace std;

 

GLfloat KEY_DEPTH_Z=2;//更改z坐标，看看有啥变化
GLfloat KEY_ZROTATE_R=0;//旋转Z轴
bool KEY_SHOW_AXIS=true;

int INTERVAL=100;
bool rotate=0;

Mesh meshobj("data/garg.obj");
Camera camera;



//https://www.opengl.org/resources/libraries/glut/spec3/spec3.html
//https://www.khronos.org/registry/OpenGL-Refpages/gl4/
void init(int argc,char **argv){
    

    int top=30,left=30,width=800,height=600;    
    // 第一步窗口的初始化等
    glutInitWindowPosition(left,top);
    glutInitWindowSize(width,height);

    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
    glutInit(&argc,argv);
    
    glutCreateWindow("Hello Computer Graphics");
    //很重要，而且要放对位置，不然后面的物体会遮挡前面的物体
   glEnable(GL_DEPTH_TEST);   // Depth testing must be turned on
   glEnable(GL_LIGHTING);     // Enable lighting calculations
   glEnable(GL_LIGHT0);       // Turn on light #0.

}


//画x,y,z三个轴

void drawAxis(){
    GLfloat LINEWIDTH=2;
    drawLines({{0,0,0},{1,0,0}},LINEWIDTH);
    drawLines({{0,0,0},{0,1,0}},LINEWIDTH);
    drawLines({{0,0,0},{0,0,1}},LINEWIDTH);
}

void drawAxis1(){
    GLfloat LINEWIDTH=5;
    drawLines({{0,0,0},{1,0,0}},LINEWIDTH);
    drawLines({{0,0,0},{0,1,0}},LINEWIDTH);
    drawLines({{0,0,0},{0,0,1}},LINEWIDTH);
}

//设置3个帧。后2个是平移第一个得到的，然后选择，用于展示 继承结构 
void example2(){
    glPushMatrix();
        glRotated(KEY_ZROTATE_R,0,0,1);
        drawAxis();
        //复制一个axis,我希望它同时旋转,push起到隔离的作用！
        glPushMatrix();
            glTranslated(.4,.4,0);
            drawAxis();
        glPopMatrix();
        //复制一个axis,我希望它同时旋转，这里的push,pop可有可无
        // glPushMatrix();
            glTranslated(-.4,-.4,0);
            drawAxis();
        // glPopMatrix();

    glPopMatrix();
}
//展示glutSolidCube,glutSolidSphere用法
void example3(){
    // glRotated(KEY_ZROTATE_R,0,1,0);
    glPushMatrix();
        //下面表示先scale 再移动，越接近模型函数的操作越先被执行！
        glTranslated(0.5,0.5,0);
        glScalef(0.25,0.25,0.25);
        
        
        
        
        //中心是0,0,0,边长是1
        glutSolidCube(1.0);
    glPopMatrix();

    glPushMatrix();
        // Matrix4f m;
        // glLoadMatrixf(m);
        // glGetFloatv(GL_MATRIX_MODE,m);
        // cout<<"xx"<<endl;
        // m.print();
        // glLoadMatrixd(&m);
        // glTranslated(-1,-1,0);
        
        glutSolidSphere(0.125,12,12);
    glPopMatrix();
}

void example4(){
    Matrix4f m=Matrix4f::rotateZ(3.1415926/6);
    m.print();
    cout<<endl;
    glLoadMatrixf(m);
    // glLoadMatrixf(Matrix4f::identity());
    drawAxis1();

    m=Matrix4f::rotateZ(-3.1415926/6);
    m=m*m.uniformScaling(0.2);
    glLoadMatrixf(m);
    drawAxis1();
}

void display(){
    glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    
    glLoadMatrixf(camera.getViewMatrix());


    
    GLfloat Lt0diff[] = {1.0,1.0,1.0,1.0};
    
    // Light position
	GLfloat Lt0pos[] = {1.0f, 1.0f, 5.0f, 1.0f};
    
    glLightfv(GL_LIGHT0,GL_DIFFUSE,Lt0diff);
    // glLightfv(GL_LIGHT0,GL_AMBIENT,Lt0diff);
    glLightfv(GL_LIGHT0,GL_POSITION,Lt0pos);
    // if(KEY_SHOW_AXIS) drawAxis();
    meshobj.draw();
 

    // glutSolidTeapot(0.6);
    
    
    glutSwapBuffers();
}
void reshape(int w,int h){  
    camera.setDimension(w,h);
    camera.perspective_projection(30,(float)w/float(h),1.f,100.f);
    //(0,0)在左下角
    glViewport(0,0,w,h);
}
void keyFunc(unsigned char key,int x, int y){
    switch (key)
    {
    case 'q':
        // exit(0);
        camera.setEyePoint({0,0,10});
        break;
   
    case 'Z':
        KEY_DEPTH_Z+=1;
        break;
    case 'z':
        KEY_DEPTH_Z-=1;
        break;
    case 'R':
        KEY_ZROTATE_R+=1;
        break;
    case 'r':
        KEY_ZROTATE_R-=1;
        break;
    case 'p':
        KEY_SHOW_AXIS=!KEY_SHOW_AXIS;
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
    // cout<<"Button:"<<button<<endl;
    // cout<<"mouseFunc:("<<x<<","<<y<<")"<<endl;
    // cout<<"state:"<<state<<endl;;
    camera.mouseFunc(button,state,x,y);
}
void motionFunc(int x, int y){
    camera.motionFunc(x,y);
}
void menu_hander(int menu){
    if(menu==3)
        meshobj.setWired(true);
    if(menu==4)
        meshobj.setWired(false);
    glutPostRedisplay();
}
void init_menu(){
    glutPostRedisplay();
}
int main(int argc,char **argv){
    init(argc,argv);
    // init_menu();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyFunc);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    //glutTimerFunc(INTERVAL,timefunc,1);
    
    // createControlWidge();
    glutMainLoop();
    return 0;
}
