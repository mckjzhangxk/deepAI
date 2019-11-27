#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Object3D.h"
#include <vecmath.h>
#include <cmath>
#include <iostream>
#include "draw.h"

using namespace std;
///TODO: implement this class.
///Add more fields as necessary,
///but do not remove hasTex, normals or texCoords
///they are filled in by other components
class Triangle: public Object3D
{
public:
	Triangle();
        ///@param a b c are three vertex positions of the triangle
        Triangle( const Vector3f& a, const Vector3f& b, const Vector3f& c);
        Vector3f compute_norm();
        void setTextCoords(Vector2f ts[3]);
        void setNormals(Vector3f ts[3]);
private:
	Vector3f m_a;
	Vector3f m_b;
	Vector3f m_c;


        bool hasTex;
        Vector3f m_normals[3];
        Vector2f m_texCoords[3];

        // Object3D interface
public:
        void draw(bool);
};

#endif //TRIANGLE_H
