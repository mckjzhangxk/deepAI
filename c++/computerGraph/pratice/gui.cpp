#include "gui.h"
void popUpMenu(void (* main_menu_hander)(int) ,void (* custom_menu_hander)(int)){
    int mainmenu=glutCreateMenu(main_menu_hander);
    int mymenu=glutCreateMenu(custom_menu_hander);
    
    glutSetMenu(mainmenu);
    glutAddMenuEntry("FullScreen Mode",1);
    glutAddMenuEntry("Window Mode",2);
    glutAddMenuEntry("Wireframe",3);
    glutAddMenuEntry("surface",4);
    glutAddMenuEntry("show frames",5);

    glutAddMenuEntry("----------------",0);

    glutAttachMenu(GLUT_RIGHT_BUTTON);
    glutAddSubMenu("Function",mymenu);
    glutSetMenu(mymenu);

    glutAddMenuEntry("z=x*y",1);
    glutAddMenuEntry("z=x+y",2);
}
