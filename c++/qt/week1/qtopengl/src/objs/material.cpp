#include "material.h"

Material::Material():m_diffuse(1.f),m_ambient(0.f),m_specular(1.f),m_shiness(1.f)
{

}

void Material::setDiffuse(Vector4f diffuse)
{
    m_diffuse=diffuse;
}

void Material::setAmbient(Vector4f ambient)
{
    m_ambient=ambient;
}

void Material::setSpecular(Vector4f specular)
{
    m_specular=specular;
}

void Material::setShiness(float shine)
{
    m_shiness=shine;
}

void Material::loadMaterial()
{

     glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,m_ambient);
     glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,m_diffuse);
     glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,m_specular);
     glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,m_shiness);
}
