#include "Object3D.h"


Object3D::Object3D():m_material(nullptr)
{

}

void Object3D::set_material(Material* v)
{
    m_material=v;
}
