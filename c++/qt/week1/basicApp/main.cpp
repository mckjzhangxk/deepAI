#include <iostream>
#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QHBoxLayout>

using namespace std;

int main(int argc,char* argv[])
{
    QApplication app(argc,argv);
    QWidget window;
//    window.resize(300,300);

    QHBoxLayout hlayout;
    QPushButton bn1("close");
    QPushButton bn2("open");
    QPushButton bn3("file");

    hlayout.addWidget(&bn1);
    hlayout.addWidget(&bn2);
    hlayout.addWidget(&bn3);

    window.setLayout(&hlayout);

    window.show();
    return app.exec();
}
