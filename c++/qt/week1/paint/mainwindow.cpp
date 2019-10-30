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
    QPen pointPen(Qt::black);
    QPen linePen(Qt::red);
    pointPen.setWidth(8);
    linePen.setWidth(2);

    QPoint p1(100,100);
    QPoint p2(200,300);

    painter.setPen(linePen);
    painter.drawLine(p1,p2);
    linePen.setJoinStyle(Qt::MiterJoin);

    painter.setPen(pointPen);
    painter.drawPoint(p1);
    painter.drawPoint(p2);

    QRect rect(p1,p2);
    QPen framePen(Qt::blue);
    QBrush brush(Qt::red,Qt::CrossPattern);


    painter.setPen(framePen);

    painter.fillRect(rect,brush);

    painter.drawRect(rect);
    painter.drawEllipse(rect);


    QPolygon poly;



    poly<<QPoint(30,30);
    poly<<QPoint(30,130);
    poly<<QPoint(130,30);
    poly<<QPoint(130,130);

     QPainterPath path;
     path.addPolygon(poly);

    painter.drawPolygon(poly);
    painter.fillPath(path,brush);

}
