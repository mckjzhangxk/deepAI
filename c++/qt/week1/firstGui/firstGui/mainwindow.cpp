#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->pushButton->setText("CLose");

    //SIGNAL,SLOT is macro, but how it's work?
    connect(ui->horizontalSlider,SIGNAL(valueChanged(int)),ui->verticalSlider,SLOT(setValue(int)));
}

MainWindow::~MainWindow()
{
    delete ui;
}

