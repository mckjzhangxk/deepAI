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
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DialogMaterial
{
public:
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QSlider *shiness;
    QWidget *widget;
    QVBoxLayout *verticalLayout;
    QSlider *diffuse_r;
    QSlider *diffuse_g;
    QSlider *diffuse_b;
    QWidget *widget1;
    QVBoxLayout *verticalLayout_2;
    QSlider *specular_r;
    QSlider *specular_g;
    QSlider *specular_b;

    void setupUi(QDialog *DialogMaterial)
    {
        if (DialogMaterial->objectName().isEmpty())
            DialogMaterial->setObjectName(QString::fromUtf8("DialogMaterial"));
        DialogMaterial->resize(410, 414);
        label = new QLabel(DialogMaterial);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(10, 20, 67, 17));
        label_2 = new QLabel(DialogMaterial);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(10, 140, 67, 17));
        label_3 = new QLabel(DialogMaterial);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(10, 270, 67, 17));
        shiness = new QSlider(DialogMaterial);
        shiness->setObjectName(QString::fromUtf8("shiness"));
        shiness->setGeometry(QRect(90, 270, 281, 16));
        shiness->setValue(50);
        shiness->setOrientation(Qt::Horizontal);
        widget = new QWidget(DialogMaterial);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setGeometry(QRect(90, 10, 291, 91));
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        diffuse_r = new QSlider(widget);
        diffuse_r->setObjectName(QString::fromUtf8("diffuse_r"));
        diffuse_r->setMaximum(100);
        diffuse_r->setSingleStep(1);
        diffuse_r->setValue(50);
        diffuse_r->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(diffuse_r);

        diffuse_g = new QSlider(widget);
        diffuse_g->setObjectName(QString::fromUtf8("diffuse_g"));
        diffuse_g->setValue(50);
        diffuse_g->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(diffuse_g);

        diffuse_b = new QSlider(widget);
        diffuse_b->setObjectName(QString::fromUtf8("diffuse_b"));
        diffuse_b->setValue(50);
        diffuse_b->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(diffuse_b);

        widget1 = new QWidget(DialogMaterial);
        widget1->setObjectName(QString::fromUtf8("widget1"));
        widget1->setGeometry(QRect(90, 120, 291, 111));
        verticalLayout_2 = new QVBoxLayout(widget1);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        specular_r = new QSlider(widget1);
        specular_r->setObjectName(QString::fromUtf8("specular_r"));
        specular_r->setValue(50);
        specular_r->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(specular_r);

        specular_g = new QSlider(widget1);
        specular_g->setObjectName(QString::fromUtf8("specular_g"));
        specular_g->setValue(50);
        specular_g->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(specular_g);

        specular_b = new QSlider(widget1);
        specular_b->setObjectName(QString::fromUtf8("specular_b"));
        specular_b->setValue(50);
        specular_b->setOrientation(Qt::Horizontal);

        verticalLayout_2->addWidget(specular_b);


        retranslateUi(DialogMaterial);

        QMetaObject::connectSlotsByName(DialogMaterial);
    } // setupUi

    void retranslateUi(QDialog *DialogMaterial)
    {
        DialogMaterial->setWindowTitle(QApplication::translate("DialogMaterial", "Dialog", nullptr));
        label->setText(QApplication::translate("DialogMaterial", "Diffuse:", nullptr));
        label_2->setText(QApplication::translate("DialogMaterial", "Specular:", nullptr));
        label_3->setText(QApplication::translate("DialogMaterial", "shiness:", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DialogMaterial: public Ui_DialogMaterial {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOGMATERIAL_H
