#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::paintEvent(QPaintEvent *event){
    QPainter painter(this);
    QTextDocument doc;
    QString html="<ul>     <li>Fundamentals of computer graphics algorithms</li>     <li>Basics of real-time rendering and graphics hardware</li>     <li>Basic OpenGL</li>     <li>C++ programming experience</li> </ul>";
    QRect rect(0,0,300,300);
    painter.translate(100,100);
    doc.setHtml(html);
    doc.drawContents(&painter,rect);


}
