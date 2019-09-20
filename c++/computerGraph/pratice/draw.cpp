#include "draw.h"

void drawPoint(const Vector3f &point,const Vector3f& normal){
        glVertex3d(point.x(),point.y(),point.z());
        glNormal3d(normal.x(),normal.y(),normal.z());        
}
void drawPoint(const Vector3f &point){
        glVertex3d(point.x(),point.y(),point.z());     
}

void drawTriangle(const vector<vector<Vector3f> > pts,const Vector3f &color){
    
    glBegin(GL_TRIANGLES);
        glColor3d(color.x(),color.y(),color.z());
        for(int i=0;i<pts.size();i++){
            drawPoint(pts[i][0],pts[i][1]);
        }
    glEnd();
}
void drawLines(const vector<Vector3f> pts,const Vector3f &color,GLfloat linewidth){
    glLineWidth(linewidth);//放在begin里面不起作用
    /*
    GL_LINES:每2个点组成一条线
    GL_LINE_STRIP:一个点连接下一个点
    */
    glBegin(GL_LINES);    
        glColor3d(color.x(),color.y(),color.z());
        for(int i=0;i<pts.size();i++){
            drawPoint(pts[i]);
        }
    glEnd();
}