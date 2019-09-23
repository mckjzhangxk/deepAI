#include<iostream>
#include<sstream>
#include<GL/glut.h>
#include<memory.h>
#include<string>
#include<vecmath/Matrix4f.h>
#include<vecmath/Vector3f.h>
#include<vector>
#include "draw.h"

using namespace std;

 


GLfloat KEY_VIEW_ANGLE_A=30;//视角，Y方向
GLfloat KEY_VIEW_ASPECT_V=1.7;//视图的w/h,
GLint KEY_OFFSET_O=0;//viewport xmin,ymin
GLfloat KEY_DEPTH_Z=2;//更改z坐标，看看有啥变化
GLfloat KEY_ZROTATE_R=0;//旋转Z轴
bool KEY_SHOW_AXIS=true;

int INTERVAL=100;
bool rotate=0;

void timefunc(int value){

    
    glutTimerFunc(INTERVAL,timefunc,value);
}


//https://www.opengl.org/resources/libraries/glut/spec3/spec3.html
//https://www.khronos.org/registry/OpenGL-Refpages/gl4/
void init(int argc,char **argv){
    

    int top=30,left=30,width=1024,height=600;    
    // 第一步窗口的初始化等
    glutInitWindowPosition(left,top);
    glutInitWindowSize(width,height);

    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
    glutInit(&argc,argv);
    
    glutCreateWindow("Hello Computer Graphics");
    //很重要，而且要放对位置，不然后面的物体会遮挡前面的物体
   glEnable(GL_DEPTH_TEST);   // Depth testing must be turned on
}

//画3个三角形
void example1(){
    drawTriangle({
        {{0,0,0},{0,0,1}},
        {{1,0,0},{0,0,1}},
        {{1,1,0},{0,0,1}},
    },{1,0,0});

    drawTriangle({
        {{0,0,0},{0,0,1}},
        {{-1,0,0},{0,0,1}},
        {{-1,-1,0},{0,0,1}},
    },{0,1,0});
   
     drawTriangle({
        {{0,2,KEY_DEPTH_Z},{0,0,1}},
        {{-1,-1,KEY_DEPTH_Z},{0,0,1}},
        {{1,-1,KEY_DEPTH_Z},{0,0,1}},
    },{0,0,1}); 

    // drawTriangle({
    //     {{0,0,0},{0,0,1}},
    //     {{1,0,0},{0,0,1}},
    //     {{1,1,0},{0,0,1}},
    //     {{0,0,0},{0,0,1}},
    //     {{-1,0,0},{0,0,1}},
    //     {{-1,-1,0},{0,0,1}},
    //     {{0,2,KEY_Z},{0,0,1}},
    //     {{-1,-1,KEY_Z},{0,0,1}},
    //     {{1,-1,KEY_Z},{0,0,1}},
    // }
    // ,{0,1,0});
}
//画x,y,z三个轴

void drawAxis(){
    GLfloat LINEWIDTH=2;
    drawLines({{0,0,0},{1,0,0}},{1,0,0},LINEWIDTH);
    drawLines({{0,0,0},{0,1,0}},{0,1,0},LINEWIDTH);
    drawLines({{0,0,0},{0,0,1}},{0,0,1},LINEWIDTH);
}

void drawAxis1(){
    GLfloat LINEWIDTH=5;
    drawLines({{0,0,0},{1,0,0}},{1,1,0},LINEWIDTH);
    drawLines({{0,0,0},{0,1,0}},{0,1,1},LINEWIDTH);
    drawLines({{0,0,0},{0,0,1}},{1,0,1},LINEWIDTH);
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
void setPespectiveView(){
    Matrix4f I=Matrix4f::identity();
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(I);
    gluPerspective(KEY_VIEW_ANGLE_A,KEY_VIEW_ASPECT_V, 1.0, 100.0);

}
void display(){
    glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
    gluLookAt(0,0,2,0,0,0,0,1,0);
    glMatrixMode(GL_MODELVIEW);
    //  glLoadIdentity();
    glLoadMatrixf(Matrix4f::identity());
    
    
    if(KEY_SHOW_AXIS) drawAxis();
    example4();
    
    
    setPespectiveView();
    glutSwapBuffers();
}
void reshape(int w,int h){  
    //(0,0)在左下角
    glViewport(KEY_OFFSET_O,KEY_OFFSET_O,w-KEY_OFFSET_O,h-KEY_OFFSET_O);
   
}
void keyFunc(unsigned char key,int x, int y){
    switch (key)
    {
    case 'q':
        exit(0);
        break;
    case 'V':
        KEY_VIEW_ASPECT_V+=0.1;
        break;
    case 'v':
        KEY_VIEW_ASPECT_V-=0.1;
        break;
    case 'A':
        KEY_VIEW_ANGLE_A+=1;
        break;
    case 'a':
        KEY_VIEW_ANGLE_A-=1;
        break;
    case 'O':
        KEY_OFFSET_O+=1;
        reshape(glutGet(GLUT_WINDOW_WIDTH),glutGet(GLUT_WINDOW_HEIGHT));
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
    cout<<(button==GLUT_LEFT_BUTTON)<<endl;
    cout<<"mouseFunc:("<<x<<","<<y<<")"<<endl;
    cout<<(state==GLUT_DOWN)<<endl;GLUT_DOWN;
}
int main(int argc,char **argv){
    init(argc,argv);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyFunc);
    glutMouseFunc(mouseFunc);
    //glutTimerFunc(INTERVAL,timefunc,1);
    glutMainLoop();
    return 0;
}
