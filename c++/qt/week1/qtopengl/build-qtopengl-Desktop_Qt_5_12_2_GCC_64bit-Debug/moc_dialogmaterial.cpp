/****************************************************************************
** Meta object code from reading C++ file 'dialogmaterial.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/view/dialogmaterial.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'dialogmaterial.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_DialogMaterial_t {
    QByteArrayData data[19];
    char stringdata0[399];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DialogMaterial_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DialogMaterial_t qt_meta_stringdata_DialogMaterial = {
    {
QT_MOC_LITERAL(0, 0, 14), // "DialogMaterial"
QT_MOC_LITERAL(1, 15, 25), // "on_diffuse_r_valueChanged"
QT_MOC_LITERAL(2, 41, 0), // ""
QT_MOC_LITERAL(3, 42, 5), // "value"
QT_MOC_LITERAL(4, 48, 25), // "on_diffuse_g_valueChanged"
QT_MOC_LITERAL(5, 74, 25), // "on_diffuse_b_valueChanged"
QT_MOC_LITERAL(6, 100, 10), // "updateView"
QT_MOC_LITERAL(7, 111, 26), // "on_specular_r_valueChanged"
QT_MOC_LITERAL(8, 138, 26), // "on_specular_g_valueChanged"
QT_MOC_LITERAL(9, 165, 26), // "on_specular_b_valueChanged"
QT_MOC_LITERAL(10, 192, 23), // "on_shiness_valueChanged"
QT_MOC_LITERAL(11, 216, 23), // "on_light_r_valueChanged"
QT_MOC_LITERAL(12, 240, 23), // "on_light_g_valueChanged"
QT_MOC_LITERAL(13, 264, 23), // "on_light_b_valueChanged"
QT_MOC_LITERAL(14, 288, 27), // "on_light_pos_x_valueChanged"
QT_MOC_LITERAL(15, 316, 4), // "arg1"
QT_MOC_LITERAL(16, 321, 27), // "on_light_pos_y_valueChanged"
QT_MOC_LITERAL(17, 349, 27), // "on_light_pos_z_valueChanged"
QT_MOC_LITERAL(18, 377, 21) // "on_toolButton_clicked"

    },
    "DialogMaterial\0on_diffuse_r_valueChanged\0"
    "\0value\0on_diffuse_g_valueChanged\0"
    "on_diffuse_b_valueChanged\0updateView\0"
    "on_specular_r_valueChanged\0"
    "on_specular_g_valueChanged\0"
    "on_specular_b_valueChanged\0"
    "on_shiness_valueChanged\0on_light_r_valueChanged\0"
    "on_light_g_valueChanged\0on_light_b_valueChanged\0"
    "on_light_pos_x_valueChanged\0arg1\0"
    "on_light_pos_y_valueChanged\0"
    "on_light_pos_z_valueChanged\0"
    "on_toolButton_clicked"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DialogMaterial[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      15,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   89,    2, 0x08 /* Private */,
       4,    1,   92,    2, 0x08 /* Private */,
       5,    1,   95,    2, 0x08 /* Private */,
       6,    0,   98,    2, 0x08 /* Private */,
       7,    1,   99,    2, 0x08 /* Private */,
       8,    1,  102,    2, 0x08 /* Private */,
       9,    1,  105,    2, 0x08 /* Private */,
      10,    1,  108,    2, 0x08 /* Private */,
      11,    1,  111,    2, 0x08 /* Private */,
      12,    1,  114,    2, 0x08 /* Private */,
      13,    1,  117,    2, 0x08 /* Private */,
      14,    1,  120,    2, 0x08 /* Private */,
      16,    1,  123,    2, 0x08 /* Private */,
      17,    1,  126,    2, 0x08 /* Private */,
      18,    0,  129,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Double,   15,
    QMetaType::Void, QMetaType::Double,   15,
    QMetaType::Void, QMetaType::Double,   15,
    QMetaType::Void,

       0        // eod
};

void DialogMaterial::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<DialogMaterial *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->on_diffuse_r_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->on_diffuse_g_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->on_diffuse_b_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->updateView(); break;
        case 4: _t->on_specular_r_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->on_specular_g_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->on_specular_b_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->on_shiness_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->on_light_r_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->on_light_g_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->on_light_b_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->on_light_pos_x_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 12: _t->on_light_pos_y_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 13: _t->on_light_pos_z_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 14: _t->on_toolButton_clicked(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject DialogMaterial::staticMetaObject = { {
    &QDialog::staticMetaObject,
    qt_meta_stringdata_DialogMaterial.data,
    qt_meta_data_DialogMaterial,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *DialogMaterial::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DialogMaterial::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DialogMaterial.stringdata0))
        return static_cast<void*>(this);
    return QDialog::qt_metacast(_clname);
}

int DialogMaterial::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 15)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 15;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 15)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 15;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
