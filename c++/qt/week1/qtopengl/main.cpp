#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWidth(1024);
    w.setHeight(768);

    w.show();
    return a.exec();
}
