#include"extgraph.h"

static const int WIDTH=1024;
static const int HEIGHT=768;
static GWindow* win=nullptr;
static string g_color="";
static GPoint g_point;

void InitGraphics(){
    if(!win){
        win=new GWindow(WIDTH,HEIGHT);
        win->setTitle("Maze Game");
    }
    win->center();
    win->clearCanvasPixels();
}
void UpdateDisplay(){

}
void SetPenColor(string color){
   g_color=color;
}
void MovePen(double x,double y){
    g_point=GPoint(x,win->getHeight()-y);
}
void DrawLine(double x,double y){
    win->setColor(g_color);
    GPoint current(g_point.getX()+x,g_point.getY()-y);
    win->drawLine(g_point,current);
    g_point=current;
}
double GetWindowWidth(){
    return WIDTH;
}
double GetWindowHeight(){
    return HEIGHT;
}



