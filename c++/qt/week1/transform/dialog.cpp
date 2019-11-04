#include "dialog.h"
#include "./ui_dialog.h"

Dialog::Dialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::Dialog)
{
    ui->setupUi(this);
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::paintEvent(QPaintEvent *event){
    QPainter painter(this);

    painter.resetMatrix();

    QRect rect(100,100,400,400);
    QPen pen1(Qt::black);
    pen1.setWidth(2);

    painter.setPen(pen1);
    painter.drawRect(rect);

    //rotate around center

    pen1.setColor(Qt::red);
    painter.resetTransform();

    painter.translate(rect.center());
    painter.rotate(45);
    painter.translate(-rect.center());

    for(int i=1;i<=10;i++){
        painter.resetTransform();
        painter.translate(rect.center());
        painter.scale(pow(0.9,i),pow(0.9,i));
        painter.rotate(45);
        qDebug()<<pow(0.9,i);
        painter.translate(-rect.center());
        painter.drawRect(rect);
    }

}
