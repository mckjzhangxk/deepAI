#include"gwindow.h"
#include "simpio.h"
#include <iostream>
#include "vector.h"
#include "random.h"
#include <cmath>
using namespace std;
static GWindow* win;

double mysqrt(double x){
    return sin(2*3.1415976*x/20)*50;
}
void draw(int x,int y,double start,double end,double(fn)(double)){
    win->setColor("red");
    for(double i=start;i<end;i+=0.1){
        win->drawPixel(x+i,y-fn(i));

    }
}

void plot(){
    win=new GWindow(800,600);
    win->setTitle("plot function");
    win->setLocation(50,50);
    const int X=0;
    const int Y=300;
    draw(X,Y,0,600,mysqrt);
}
