#ifndef OBJECT_ZXK
#define OBJECT_ZXK
#include "material.h"
class Object3D{
public:
    Object3D();
    virtual void draw(bool )=0;
    void set_material(Material* v);
protected:
    Material *m_material;
};
#endif
