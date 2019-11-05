#ifndef MATERIAL_H
#define MATERIAL_H
#include<Vector4f.h>
#include <GL/glut.h>
class Material
{
public:
    Material();
    void setDiffuse(Vector4f diffuse);
    void setAmbient(Vector4f ambient);
    void setSpecular(Vector4f specular);
    void setShiness(float shine);
    void loadMaterial();
private:
    Vector4f m_diffuse;
    Vector4f m_ambient;
    Vector4f m_specular;
    float m_shiness;
};

#endif // MATERIAL_H
