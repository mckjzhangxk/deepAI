#include "dialogmaterial.h"
#include "ui_dialogmaterial.h"

DialogMaterial::DialogMaterial(QWidget *parent):
    QDialog(parent),
    ui(new Ui::DialogMaterial),
    m_window(nullptr,0,false)
{
    ui->setupUi(this);
    m_material=Material();

    m_mesh=new Mesh("/home/zhangxk/AIProject/pytorch_coma/template/template.obj");
    m_mesh->set_material(&m_material);


    m_window.setMeshObj(m_mesh);
    m_window.setLight(&m_light);

    m_window.show();
    updateView();
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


    r=(float( ui->light_r->value()))/100.;
    g=(float( ui->light_g->value()))/100.;
    b=(float( ui->light_b->value()))/100.;
    m_light.setColor({r,g,b,1.f});

    float x=ui->light_pos_x->value();
    float y=ui->light_pos_y->value();
    float z=ui->light_pos_z->value();
    m_light.setPosition({x,y,z,1.f});

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

void DialogMaterial::on_light_r_valueChanged(int value)
{
    updateView();
}

void DialogMaterial::on_light_g_valueChanged(int value)
{
    updateView();
}

void DialogMaterial::on_light_b_valueChanged(int value)
{
    updateView();
}

void DialogMaterial::on_light_pos_x_valueChanged(double arg1)
{
    updateView();
}

void DialogMaterial::on_light_pos_y_valueChanged(double arg1)
{
        updateView();
}

void DialogMaterial::on_light_pos_z_valueChanged(double arg1)
{
        updateView();
}

void DialogMaterial::on_toolButton_clicked()
{
    QFileDialog d;
    QString filename=d.getOpenFileName();
    if(filename!=""){
        char * fc=filename.toLatin1().data();
        qDebug()<<fc;
        m_mesh->loadMesh(fc);
        updateView();
    }
}
