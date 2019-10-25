#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QOpenGLWindow>

class MainWindow : public QOpenGLWindow
{
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // QWindow interface


    // QOpenGLWindow interface
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *);
};
#endif // MAINWINDOW_H
