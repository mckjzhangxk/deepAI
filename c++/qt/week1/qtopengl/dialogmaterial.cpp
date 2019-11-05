#include "dialogmaterial.h"
#include "ui_dialogmaterial.h"

DialogMaterial::DialogMaterial(QWidget *parent):
    QDialog(parent),
    ui(new Ui::DialogMaterial),
    m_window(nullptr,0,false)
{
    ui->setupUi(this);
    m_mesh=new Mesh("/home/zhangxk/projects/deepAI/c++/qt/week1/qtopengl/data/garg.obj");
    m_material=Material();
    m_mesh->set_material(&m_material);
    m_window.setMeshObj(m_mesh);


    m_window.show();
}

DialogMaterial::~DialogMaterial()
{
    delete ui;
}

void DialogMaterial::on_diffuse_r_valueChanged(int value)
{

    updateView();
}

void DialogMaterial::on_diffuse_g_valueChanged(int value)
{

    updateView();
}

void DialogMaterial::on_diffuse_b_valueChanged(int value)
{

    updateView();
}

void DialogMaterial::updateView()
{
    float r=(float( ui->diffuse_r->value()))/100.;
    float g=(float( ui->diffuse_g->value()))/100.;
    float b=(float( ui->diffuse_b->value()))/100.;
    m_material.setDiffuse({r,g,b,1.f});


    r=(float( ui->specular_r->value()))/100.;
    g=(float( ui->specular_g->value()))/100.;
    b=(float( ui->specular_b->value()))/100.;
    m_material.setSpecular({r,g,b,1.f});

    m_material.setShiness(ui->shiness->value());
    m_window.reflesh();
}

void DialogMaterial::on_specular_r_valueChanged(int value)
{
    updateView();
}

void DialogMaterial::on_specular_g_valueChanged(int value)
{
    updateView();
}

void DialogMaterial::on_specular_b_valueChanged(int value)
{
    updateView();
}


void DialogMaterial::on_shiness_valueChanged(int value)
{
     updateView();
}
