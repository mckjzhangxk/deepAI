#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <GL/glut.h>
#include "mesh.h"
#include "Camera.h"
#include <qgl.h>
#include<QKeyEvent>
#include <QMouseEvent>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QGLWidget
{
    Q_OBJECT

public:
    MainWindow( QWidget* parent = 0, const char* name = 0, bool fs = false );
    ~MainWindow();
    void setMeshObj(Mesh* v);
    void reflesh();
private:
    // QPaintDeviceWindow interface
    bool fullscreen;
    Mesh* m_mesh;
    Camera m_camera;
    Ui::MainWindow *ui;
protected:
    virtual void initializeGL() override;
    virtual void resizeGL(int w, int h) override;
    virtual void paintGL() override;
    virtual void keyPressEvent(QKeyEvent *event) override;


    // QWidget interface
protected:
    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void mouseReleaseEvent(QMouseEvent *event) override;
    virtual void mouseMoveEvent(QMouseEvent *event) override;

    // QWidget interface
protected:
    virtual void wheelEvent(QWheelEvent *event) override;
};


#endif // MAINWINDOW_H
