#ifndef DIALOGMATERIAL_H
#define DIALOGMATERIAL_H

#include <QDialog>
#include <QFileDialog>
#include <Vector4f.h>
#include "mainwindow.h"
#include "mesh.h"
#include "light.h"
#include <QDebug>
namespace Ui {
class DialogMaterial;
}

class DialogMaterial : public QDialog
{
    Q_OBJECT

public:
    explicit DialogMaterial(QWidget *parent = nullptr);
    ~DialogMaterial();

private slots:


    void on_diffuse_r_valueChanged(int value);

    void on_diffuse_g_valueChanged(int value);

    void on_diffuse_b_valueChanged(int value);

    void updateView();
    void on_specular_r_valueChanged(int value);

    void on_specular_g_valueChanged(int value);

    void on_specular_b_valueChanged(int value);


    void on_shiness_valueChanged(int value);

    void on_light_r_valueChanged(int value);

    void on_light_g_valueChanged(int value);

    void on_light_b_valueChanged(int value);

    void on_light_pos_x_valueChanged(double arg1);

    void on_light_pos_y_valueChanged(double arg1);

    void on_light_pos_z_valueChanged(double arg1);

    void on_toolButton_clicked();

private:
    Ui::DialogMaterial *ui;    
    Mesh* m_mesh;
    Material m_material;
    Light m_light;
    MainWindow m_window;
};

#endif // DIALOGMATERIAL_H
