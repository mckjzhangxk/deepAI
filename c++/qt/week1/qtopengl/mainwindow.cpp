#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
{

    setSurfaceType(QWindow::OpenGLSurface);
    QSurfaceFormat format;
    format.setProfile(QSurfaceFormat::CompatibilityProfile);
    format.setVersion(2,1);
    setFormat(format);

//    m_context=new QOpenGLContext();
//    m_context->setFormat(format);
//    m_context->create();
//    m_context->makeCurrent(this);

//    m_functions=m_context->functions();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    paintGL();
}

void MainWindow::initializeGL()
{
    //很重要，而且要放对位置，不然后面的物体会遮挡前面的物体
   glEnable(GL_DEPTH_TEST);   // Depth testing must be turned on
   glEnable(GL_LIGHTING);     // Enable lighting calculations
   glEnable(GL_LIGHT0);       // Turn on light #0.
}

void MainWindow::resizeGL(int w, int h)
{

}

void MainWindow::paintGL()
{

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glBegin(GL_TRIANGLES);
    glVertex2f(0,0);
    glColor3f(1,0,0);
    glVertex2f(.5,0);
    glColor3f(0,1,0);
    glVertex2f(.5,.3);
    glColor3f(0,0,1);
   glEnd();
   glFlush();
}

void MainWindow::resizeEvent(QResizeEvent *)
{

}

