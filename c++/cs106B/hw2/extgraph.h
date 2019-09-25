#ifndef EXTGRAPH_H
#define EXTGRAPH_H
#include <iostream>
#include "gwindow.h"
using namespace std;
void InitGraphics();
void UpdateDisplay();
void SetPenColor(string color);
void MovePen(double x,double y);
void DrawLine(double x,double y);
double GetWindowWidth();
double GetWindowHeight();
#endif // EXTGRAPH_H
