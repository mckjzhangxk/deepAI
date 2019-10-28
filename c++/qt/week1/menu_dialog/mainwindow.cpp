#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setCentralWidget(ui->plainTextEdit);
    m_diag=new Dialog();
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_actionhelp_triggered()
{
//    m_diag->show();
     m_diag->setModal(true);
     m_diag->exec();
}
