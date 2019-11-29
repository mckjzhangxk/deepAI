#include "Triangle.h"

Triangle::Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c):hasTex(false)
{
    m_a=a;m_b=b;m_c=c;
}

Vector3f Triangle::compute_norm()
{
    Vector3f a = m_b-m_a;
    Vector3f b = m_c-m_a;
    b=Vector3f::cross(a,b);

    return b;
}

void Triangle::setTextCoords(Vector2f ts[])
{
    m_texCoords[0]=ts[0];
    m_texCoords[1]=ts[1];
    m_texCoords[2]=ts[2];
}

void Triangle::setNormals(Vector3f ts[])
{
    m_normals[0]=ts[0];
    m_normals[1]=ts[1];
    m_normals[2]=ts[2];
}

void Triangle::draw(bool wired)
{
       if(m_material){
           m_material->loadMaterial();
       }


        Vector3f& v1=m_a;
        Vector3f& v2=m_b;
        Vector3f& v3=m_c;

        Vector3f& n1=m_normals[0];
        Vector3f& n2=m_normals[1];
        Vector3f& n3=m_normals[2];

        if(wired){
            glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        }else
        {
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

        }
        drawTriangle({v1,v2,v3},{n1,n2,n3});


}

