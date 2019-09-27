#include"extgraph.h"

static const int WIDTH=800;
static const int HEIGHT=600;
static GWindow* win=nullptr;
static string g_color="";
static GPoint g_point;
static const int MARGIN_X=40;
static const int MARGIN_Y=40;
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
    g_point=GPoint(MARGIN_X+x,win->getHeight()-y);
}
void DrawLine(double x,double y){
    win->setColor(g_color);
    GPoint current(g_point.getX()+x,g_point.getY()-y);
    win->drawLine(g_point,current);
    g_point=current;
}
double GetWindowWidth(){
    return WIDTH-MARGIN_X*2;
}
double GetWindowHeight(){
    return HEIGHT-MARGIN_Y*2;
}



