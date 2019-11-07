#include "src/view/dialogmaterial.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    DialogMaterial w;

    w.show();
    return a.exec();
}
