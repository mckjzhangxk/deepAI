#include "light.h"

Light::Light():m_position(Vector4f(0,1,1,1)),m_color(1)
{

}

void Light::setup()
{
    glLightfv(GL_LIGHT0, GL_DIFFUSE, m_color);
    glLightfv(GL_LIGHT0,GL_POSITION,m_position);
}

void Light::setPosition(Vector4f v)
{
    m_position=v;
}

void Light::setColor(Vector4f v)
{
    m_color=v;
}

Vector4f Light::getPosition()
{
    return  m_position;
}

Vector4f Light::getColor()
{
    return m_color;
}
