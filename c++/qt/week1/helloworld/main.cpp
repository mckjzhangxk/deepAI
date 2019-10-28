#include <QCoreApplication>
#include <QDebug>
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QString hello="Hello";
    qDebug()<<"Hello world";
    qDebug()<<hello;
    return a.exec();
}
