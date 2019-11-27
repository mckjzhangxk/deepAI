#include "Sphere.h"

Sphere::Sphere(int radius,int clips):m_radius(radius),m_clips(clips){

}
void Sphere::draw(bool wired){
    GLfloat x, y, z, alpha, beta; // Storage for coordinates and angles
    for (alpha = 0.0; alpha < PI; alpha += PI/m_clips)
    {
        glBegin(GL_TRIANGLE_STRIP);
        for (beta = 0.0; beta < 2.01*PI; beta += PI/m_clips)
        {
            x = m_radius*cos(beta)*sin(alpha);
            y = m_radius*sin(beta)*sin(alpha);
            z = m_radius*cos(alpha);
            glVertex3f(x, y, z);
            x = m_radius*cos(beta)*sin(alpha + PI/m_clips);
            y = m_radius*sin(beta)*sin(alpha + PI/m_clips);
            z = m_radius*cos(alpha + PI/m_clips);
            glVertex3f(x, y, z);
        }
        glEnd();
    }
}

void Sphere::setRadius(float radius)
{
    m_radius=radius;
}

void Sphere::setClips(float clips)
{
    m_clips=clips;
}

