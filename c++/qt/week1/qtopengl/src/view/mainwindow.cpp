#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <GL/glut.h>
//https://qiliang.net/old/nehe_qt/lesson01.html

MainWindow::MainWindow( QWidget* parent, const char* name, bool fs ): QGLWidget( parent ),m_showwire(false)
{

  fullscreen = fs;
  setGeometry( 0, 0, 640, 480 );

  if ( fullscreen )
    showFullScreen();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setMeshObj(Mesh *v)
{
    m_mesh=v;
}

void MainWindow::setLight(Light *light)
{
    m_light=light;
}

void MainWindow::reflesh()
{
    paintGL();
    swapBuffers();
}

void MainWindow::paintGL()
{
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glMatrixMode(GL_MODELVIEW);

  glLoadMatrixf(m_camera.getViewMatrix());
  m_light->setup();
  /*glFrontFace(GL_CW);
  glCullFace(GL_FRONT);*/
  m_mesh->draw(m_showwire);
}

void MainWindow::initializeGL()
{
    glEnable(GL_DEPTH_TEST);   // Depth testing must be turned on
    glEnable(GL_LIGHTING);     // Enable lighting calculations
    glEnable(GL_LIGHT0);       // Turn on light #0.
    //glEnable(GL_CULL_FACE);
}

void MainWindow::resizeGL( int w, int h )
{

    m_camera.setDimension(w,h);
    m_camera.perspective_projection(30,(float)w/float(h),0.01f,100.f);
    //(0,0)在左下角
    glViewport(0,0,w,h);

}

void MainWindow::keyPressEvent( QKeyEvent *e )
{
  switch ( e->key() )
  {

  case Qt::Key_F2:
    fullscreen = !fullscreen;
    if ( fullscreen )
    {
      showFullScreen();
    }
    else
    {
      showNormal();
      setGeometry( 0, 0, 640, 480 );
    }
    update();
    break;
  case Qt::Key_Escape:
    close();
      break;
   case Qt::Key_W:
      m_showwire=!m_showwire;
      reflesh();
      break;
  }

}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    if(event->button()==Qt::RightButton){
        m_camera.mouseFunc(GLUT_RIGHT_BUTTON,GLUT_DOWN,event->x(),event->y());
    }
    else if(event->button()==Qt::LeftButton){
        m_camera.mouseFunc(GLUT_LEFT_BUTTON,GLUT_DOWN,event->x(),event->y());
    }
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    if(event->button()==Qt::RightButton){
        m_camera.mouseFunc(GLUT_RIGHT_BUTTON,GLUT_UP,event->x(),event->y());
    }else if(event->button()==Qt::LeftButton){
        m_camera.mouseFunc(GLUT_LEFT_BUTTON,GLUT_UP,event->x(),event->y());
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    m_camera.motionFunc(event->x(),event->y());
    reflesh();
}

void MainWindow::wheelEvent(QWheelEvent *event)
{
    if(event->angleDelta().y()>0){
        m_camera.mouseFunc(3,0,event->x(),event->y());
    }else{
        m_camera.mouseFunc(4,0,event->x(),event->y());
    }

    reflesh();

}
