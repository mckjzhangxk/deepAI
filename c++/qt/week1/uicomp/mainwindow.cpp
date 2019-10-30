#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include<QMessageBox>

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


void MainWindow::on_pushButton_clicked()
{
    ui->textEdit->clear();
    if(ui->radioButton->isChecked()){
        QMessageBox::information(this,"A","A");
    }
    if(ui->radioButton_2->isChecked()){
        QMessageBox::information(this,"B","B");
    }
}

void MainWindow::on_textEdit_textChanged()
{

}

void MainWindow::on_lineEdit_textChanged(const QString &arg1)
{
    ui->textEdit->setText(arg1);
}
