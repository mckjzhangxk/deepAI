#ifndef GUI_ZXK 
#define GUI_ZXK
#include<GL/glut.h>
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
void popUpMenu(void (* main_menu_hander)(int) ,void (* custom_menu_hander)(int));
void createControlWidge();
#endif