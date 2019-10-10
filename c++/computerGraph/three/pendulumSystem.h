#ifndef PENDULUMSYSTEM_H
#define PENDULUMSYSTEM_H

#include <vecmath.h>
#include <vector>
#include <GL/glut.h>
#include <math.h>

#include "particleSystem.h"


class PendulumSystem: public ParticleSystem
{
public:
	PendulumSystem(int numParticles);
	
	vector<Vector3f> evalF(vector<Vector3f> state);
	
	void draw();
	void reset();
private:
	float m_drag;
	float m_gravity;
	vector<vector<float> > m_springs;
};

#endif
