#include "mainwindow.h"

#include <QApplication>
#include <QDir>
#include <QDebug>
#include <QFile>
#include <QString>

void write(QString filename){
    QFile mfile(filename);
    if(!mfile.open(QFile::WriteOnly|QFile::Text)){
        qDebug()<<"can not open file";
        return;
    }
    QTextStream out(&mfile);
    out<<"First line\n";
    out<<"second line\n";

    mfile.close();
}


void read(QString filename){
    QFile mfile(filename);
    if(!mfile.open(QFile::ReadOnly|QFile::Text)){
        qDebug()<<"can not open file";
        return;
    }
    QTextStream in(&mfile);
    QString buf;

    buf=in.readLine();
    qDebug()<<buf;
    buf=in.readLine();
    qDebug()<<buf;
    mfile.close();
}
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QDir dir("/home/zhangxk");
    //A.read file list
//    for(QString s:dir.entryList()){
//        qDebug()<<s;
//    }
    //B.read or write file
//    write("my.txt");
//    read("my.txt");

    read(":/zxk/main.cpp");
    return 0;
}
