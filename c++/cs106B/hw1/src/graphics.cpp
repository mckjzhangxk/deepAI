#include"gwindow.h"
#include "simpio.h"
#include <iostream>
#include "vector.h"
#include "random.h"
using namespace  std;

static Vector<GPoint> points;
static GWindow* win;
static const string COLOR="green";
static const int CIRCLE_SIZE=4;
static const unsigned int Iters=2000;

const GPoint & randomVertex(){
    int n=randomInteger(0,points.size()-1);
    return points[n];
}
void drawTriangle(GWindow* win){
    int N=points.size();
    for(int i=0;i<points.size();i++){
        win->drawLine(points[i],points[(i+1)%N]);
    }
}
void drawCircle(GWindow* win,const GPoint & pt){
    int x=pt.getX(),y=pt.getY();
    win->drawOval(x,y,CIRCLE_SIZE,CIRCLE_SIZE);
    win->setColor(COLOR);
    win->fillOval(x,y,CIRCLE_SIZE,CIRCLE_SIZE);
}
/*
p1.x=p1.x+p2.x /2
p1.y=p1.y+p2.y /2
*/
GPoint update(const GPoint &p1,const GPoint &p2){
    double x=(p1.getX()+p2.getX())/2;
    double y=(p1.getY()+p2.getY())/2;
    return GPoint(x,y);
}
void play_chaos(){
    win->clearCanvasPixels();
    drawTriangle(win);

    GPoint currentPt=randomVertex();
    for(auto t=0;t<Iters;t++){
        const GPoint &rd_pt=randomVertex();
        currentPt=update(currentPt,rd_pt);
        drawCircle(win,currentPt);
    }
}
void hander(GEvent e){

    if(e.getType()==EventType::MOUSE_CLICKED&&e.getButton()==1){
        if(points.size()<3){
            int x=e.getX();
            int y=e.getY();
            points.add(GPoint(x,y));
            if(points.size()==3){
                play_chaos();
                points.clear();
            }
        }
    }

}
void Ghaos_Game(){
    win=new GWindow(600,600);
    win->setTitle("Ghaos_Game");
    win->setLocation(100,100);
    win->setClickListener(hander);
}
