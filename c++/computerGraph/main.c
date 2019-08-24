#include<GL/glut.h>
#include<stdio.h>

#define ROWS 10
#define COLS 30
#define NHOTS 4
#define NCOLS 5

/*

gcc main.c -lGL -lglut -lGLU
https://codeday.me/bug/20190124/558943.html
*/

GLfloat AMBIENT=25.0;
GLfloat HOT=50.0;
GLfloat COLD=0.0;
GLfloat temp[ROWS][COLS];


GLfloat angle=0.0;
GLfloat theta=0,vp=30;

int hotspots[NHOTS][2]={
    {ROWS/2,0},
    {ROWS/2-1,0},
    {ROWS/2-2,0},
    {0,3*COLS/4}
};
int coldspots[NCOLS][2]={
    {ROWS-1,COLS/3},
    {ROWS-1,1+COLS/3},
    {ROWS-1,2+COLS/3},
    {ROWS-1,3+COLS/3},
    {ROWS-1,4+COLS/3}
};
void myinit(){
    int i,j;
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.6,0.6,0.6,1.0);

    for(int i=0;i<ROWS;i++)
        for(int j=0;j<COLS;j++){
            temp[i][j]=AMBIENT;
        }
    for(int i=0;i<NHOTS;i++){
        temp[hotspots[i][0]][hotspots[i][1]]=HOT;
    }
    for(int i=0;i<NCOLS;i++){
        temp[coldspots[i][0]][coldspots[i][1]]=COLD;
    }
}

void cube(){
    typedef GLfloat point[3];

    point v[8]={
        {0,0,0},{0,0,1},
        {0,1,0},{0,1,1},
        {1,0,0},{1,0,1},
        {1,1,0},{1,1,1}
    };

    glBegin(GL_QUAD_STRIP);
        glVertex3fv(v[4]);
        glVertex3fv(v[5]);
        glVertex3fv(v[0]);
        glVertex3fv(v[1]);
        glVertex3fv(v[2]);
        glVertex3fv(v[3]);
        glVertex3fv(v[6]);
        glVertex3fv(v[7]);
    glEnd();

    glBegin(GL_QUAD_STRIP);
        glVertex3fv(v[1]);
        glVertex3fv(v[3]);
        glVertex3fv(v[5]);
        glVertex3fv(v[7]);
        glVertex3fv(v[4]);
        glVertex3fv(v[6]);
        glVertex3fv(v[0]);
        glVertex3fv(v[2]);
    glEnd();
}

void setColor(float t){
    float r,g=0;
    r=(t-COLD)/(HOT-COLD);
    glColor3f(r,g,1-r);
}

void display(){
    #define SCALE 10
    int i,j;

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(vp,vp/2,vp/4,0.0,0.0,0.0,0.0,0.0,1);

    glPushMatrix();
    glRotatef(angle,0.0,0.0,1.0);
    
    for(int i=0;i<ROWS;i++)
        for(int j=0;j<COLS;j++){
            setColor(temp[i][j]);
            glPushMatrix();
            glTranslatef((float)i-(float)ROWS/2.0,
                    (float)j-(float)COLS/2.0,0.0);
            // glScalef(1,1,0.1+3.9*temp[i][j]/HOT);
            cube();
            glPopMatrix();
        }
    glPopMatrix();
    glutSwapBuffers();
}
int main(int argc,char ** argv){

    GLfloat angle=3.0;
    // printf("%d",sizeof(GLfloat));
}
