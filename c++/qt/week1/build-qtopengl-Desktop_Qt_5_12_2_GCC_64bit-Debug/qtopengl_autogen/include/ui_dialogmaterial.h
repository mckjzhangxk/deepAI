/********************************************************************************
** Form generated from reading UI file 'dialogmaterial.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOGMATERIAL_H
#define UI_DIALOGMATERIAL_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DialogMaterial
{
public:
    QFrame *frame;
    QLabel *label_3;
    QLabel *label;
    QWidget *layoutWidget;
    QVBoxLayout *verticalLayout;
    QSlider *diffuse_r;
    QSlider *diffuse_g;
    QSlider *diffuse_b;
    QLabel *label_2;
    QSlider *shiness;
    QWidget *layoutWidget1;
    QVBoxLayout *verticalLayout_2;
    QSlider *specular_r;
    QSlider *specular_g;
    QSlider *specular_b;
    QLabel *label_4;
    QFrame *frame_2;
    QLabel *label_5;
    QLabel *label_6;
    QLabel *label_7;
    QWidget *widget;
    QVBoxLayout *verticalLayout_3;
    QSlider *light_r;
    QSlider *light_g;
    QSlider *light_b;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout;
    QDoubleSpinBox *light_pos_x;
    QDoubleSpinBox *light_pos_y;
    QDoubleSpinBox *light_pos_z;
    QToolButton *toolButton;

    void setupUi(QDialog *DialogMaterial)
    {
        if (DialogMaterial->objectName().isEmpty())
            DialogMaterial->setObjectName(QString::fromUtf8("DialogMaterial"));
        DialogMaterial->resize(686, 347);
        frame = new QFrame(DialogMaterial);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setGeometry(QRect(280, 10, 391, 291));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        label_3 = new QLabel(frame);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(20, 260, 67, 17));
        label = new QLabel(frame);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(20, 70, 67, 17));
        layoutWidget = new QWidget(frame);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(80, 60, 301, 91));
        verticalLayout = new QVBoxLayout(layoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        diffuse_r = new QSlider(layoutWidget);
        diffuse_r->setObjectName(QString::fromUtf8("diffuse_r"));
        diffuse_r->setMaximum(100);
        diffuse_r->setSingleStep(1);
        diffuse_r->setValue(50);
        diffuse_r->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(diffuse_r);

        diffuse_g = new QSlider(layoutWidget);
        diffuse_g->setObjectName(QString::fromUtf8("diffuse_g"));
        diffuse_g->setValue(50);
        diffuse_g->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(diffuse_g);

        diffuse_b = new QSlider(layoutWidget);
        diffuse_b->setObjectName(QString::fromUtf8("diffuse_b"));
        diffuse_b->setValue(50);
        diffuse_b->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(diffuse_b);

        label_2 = new QLabel(frame);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(10, 180, 67, 17));
        shiness = new QSlider(frame);
        shiness->setObjectName(QString::fromUtf8("shiness"));
        shiness->setGeometry(QRect(80, 260, 301, 16));
        shiness->setValue(50);
        shiness->setOrientation(Qt::Horizontal);
        layoutWidget1 = new QWidget(frame);
        layoutWidget1->setObjectName(QString::fromUtf8("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(80, 170, 301, 81));
        verticalLayout_2 = new QVBoxLayout(layoutWidget1);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        specular_r = new QSlider(layoutWidget1);
        specular_r->setObjectName(QString::fromUtf8("specular_r"));
        specular_r->setValue(50);
        specular_r->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(specular_r);

        specular_g = new QSlider(layoutWidget1);
        specular_g->setObjectName(QString::fromUtf8("specular_g"));
        specular_g->setValue(50);
        specular_g->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(specular_g);

        specular_b = new QSlider(layoutWidget1);
        specular_b->setObjectName(QString::fromUtf8("specular_b"));
        specular_b->setValue(50);
        specular_b->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(specular_b);

        label_4 = new QLabel(frame);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setGeometry(QRect(70, 30, 181, 17));
        frame_2 = new QFrame(DialogMaterial);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setGeometry(QRect(20, 10, 251, 291));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        label_5 = new QLabel(frame_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setGeometry(QRect(70, 20, 111, 20));
        label_6 = new QLabel(frame_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setGeometry(QRect(10, 50, 67, 17));
        label_7 = new QLabel(frame_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setGeometry(QRect(10, 130, 67, 17));
        widget = new QWidget(frame_2);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setGeometry(QRect(10, 150, 231, 81));
        verticalLayout_3 = new QVBoxLayout(widget);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        light_r = new QSlider(widget);
        light_r->setObjectName(QString::fromUtf8("light_r"));
        light_r->setValue(99);
        light_r->setOrientation(Qt::Horizontal);

        verticalLayout_3->addWidget(light_r);

        light_g = new QSlider(widget);
        light_g->setObjectName(QString::fromUtf8("light_g"));
        light_g->setValue(99);
        light_g->setOrientation(Qt::Horizontal);

        verticalLayout_3->addWidget(light_g);

        light_b = new QSlider(widget);
        light_b->setObjectName(QString::fromUtf8("light_b"));
        light_b->setValue(99);
        light_b->setOrientation(Qt::Horizontal);

        verticalLayout_3->addWidget(light_b);

        widget1 = new QWidget(frame_2);
        widget1->setObjectName(QString::fromUtf8("widget1"));
        widget1->setGeometry(QRect(10, 80, 231, 28));
        horizontalLayout = new QHBoxLayout(widget1);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        light_pos_x = new QDoubleSpinBox(widget1);
        light_pos_x->setObjectName(QString::fromUtf8("light_pos_x"));
        light_pos_x->setMinimum(-2.000000000000000);
        light_pos_x->setMaximum(2.000000000000000);
        light_pos_x->setSingleStep(0.020000000000000);

        horizontalLayout->addWidget(light_pos_x);

        light_pos_y = new QDoubleSpinBox(widget1);
        light_pos_y->setObjectName(QString::fromUtf8("light_pos_y"));
        light_pos_y->setMinimum(-2.000000000000000);
        light_pos_y->setMaximum(2.000000000000000);
        light_pos_y->setSingleStep(0.020000000000000);
        light_pos_y->setValue(1.000000000000000);

        horizontalLayout->addWidget(light_pos_y);

        light_pos_z = new QDoubleSpinBox(widget1);
        light_pos_z->setObjectName(QString::fromUtf8("light_pos_z"));
        light_pos_z->setMinimum(-2.000000000000000);
        light_pos_z->setMaximum(2.000000000000000);
        light_pos_z->setSingleStep(0.020000000000000);
        light_pos_z->setValue(1.000000000000000);

        horizontalLayout->addWidget(light_pos_z);

        toolButton = new QToolButton(DialogMaterial);
        toolButton->setObjectName(QString::fromUtf8("toolButton"));
        toolButton->setGeometry(QRect(580, 310, 89, 25));

        retranslateUi(DialogMaterial);

        QMetaObject::connectSlotsByName(DialogMaterial);
    } // setupUi

    void retranslateUi(QDialog *DialogMaterial)
    {
        DialogMaterial->setWindowTitle(QApplication::translate("DialogMaterial", "Control Pannel", nullptr));
        label_3->setText(QApplication::translate("DialogMaterial", "shiness:", nullptr));
        label->setText(QApplication::translate("DialogMaterial", "Diffuse:", nullptr));
        label_2->setText(QApplication::translate("DialogMaterial", "Specular:", nullptr));
        label_4->setText(QApplication::translate("DialogMaterial", "Material Pannel", nullptr));
        label_5->setText(QApplication::translate("DialogMaterial", "Light Pannel", nullptr));
        label_6->setText(QApplication::translate("DialogMaterial", "Position", nullptr));
        label_7->setText(QApplication::translate("DialogMaterial", "Color", nullptr));
        toolButton->setText(QApplication::translate("DialogMaterial", "model", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DialogMaterial: public Ui_DialogMaterial {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOGMATERIAL_H
