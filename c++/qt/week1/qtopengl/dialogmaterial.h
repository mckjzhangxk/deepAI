#ifndef DIALOGMATERIAL_H
#define DIALOGMATERIAL_H

#include <QDialog>
#include <Vector4f.h>
#include "mainwindow.h"
#include "mesh.h"

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

private:
    Ui::DialogMaterial *ui;
    Material m_material;
    Mesh* m_mesh;

    MainWindow m_window;
};

#endif // DIALOGMATERIAL_H
