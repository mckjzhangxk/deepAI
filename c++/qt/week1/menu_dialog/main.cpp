#include "mainwindow.h"
#include<vecmath.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    Vector2f f;
    f.print();
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
