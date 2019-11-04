#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QOpenGLWindow>
#include<QOpenGLFunctions>
#include<QSurfaceFormat>
#include <GL/glut.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QOpenGLWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QOpenGLContext *m_context;
    QOpenGLFunctions * m_functions;
    // QPaintDeviceWindow interface
protected:
    virtual void paintEvent(QPaintEvent *event) override;

    virtual void initializeGL() override;
    virtual void resizeGL(int w, int h) override;
    virtual void paintGL() override;

    virtual void resizeEvent(QResizeEvent *) override;
};


#endif // MAINWINDOW_H
