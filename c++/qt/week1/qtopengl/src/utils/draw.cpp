#include "draw.h"

void drawPoint(const Vector3f &point,const Vector3f& normal){
        glNormal3d(normal.x(),normal.y(),normal.z());
        glVertex3d(point.x(),point.y(),point.z());         
}
void drawPoint(const Vector3f &point){
        glVertex3d(point.x(),point.y(),point.z());     
}
void drawNormal(const Vector3f& norm){
    glNormal3f(norm.x(),norm.y(),norm.z());
}



void drawTriangle(const vector<Vector3f > pts,const vector<Vector3f > norms){
    assert(pts.size()==3);
    assert(norms.size()==3);

    glBegin(GL_TRIANGLES);
        for(int i=0;i<3;i++){
            drawPoint(pts[i],norms[i]);
        }
    glEnd();
}
void drawLines(const vector<Vector3f> pts,GLfloat linewidth){
    glLineWidth(linewidth);//放在begin里面不起作用
    /*
    GL_LINES:每2个点组成一条线
    GL_LINE_STRIP:一个点连接下一个点
    */
    glBegin(GL_LINES);    
        for(int i=0;i<pts.size();i++){
            drawPoint(pts[i]);
        }
    glEnd();
}
void drawLines(const vector<Vector3f> pts,const vector<Vector3f > norms,GLfloat linewidth){
    glLineWidth(linewidth);//放在begin里面不起作用
    /*
    GL_LINES:每2个点组成一条线
    GL_LINE_STRIP:一个点连接下一个点
    */
    glBegin(GL_LINES);
        for(int i=0;i<pts.size();i++){
            drawPoint(pts[i],norms[i]);
        }
    glEnd();
}
