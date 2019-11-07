#ifndef LIGHT_H
#define LIGHT_H
#include <Vector4f.h>
#include <GL/glu.h>
class Light
{
public:
    Light();
    void setup();
    void setPosition(Vector4f v);
    void setColor(Vector4f v);
    Vector4f getPosition();
    Vector4f getColor();
private:
    Vector4f m_color;
    Vector4f m_position;
};

#endif // LIGHT_H
