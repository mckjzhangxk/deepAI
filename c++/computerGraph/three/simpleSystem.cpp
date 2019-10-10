
#include "simpleSystem.h"
#include<iostream>
using namespace std;

SimpleSystem::SimpleSystem()
{
	reset();
}

// TODO: implement evalF
// for a given state, evaluate f(X,t)
vector<Vector3f> SimpleSystem::evalF(vector<Vector3f> state)
{
	vector<Vector3f> f;
	Vector3f s0=state[0];

	Vector3f dstate(-s0.y(),s0.x(),0);
	f.push_back(dstate);
	return f;
}

// render the system (ie draw the particles)
void SimpleSystem::draw()
{
		
		Vector3f pos(m_vVecState[0]) ;//YOUR PARTICLE POSITION
	  	glPushMatrix();
 
		glTranslatef(pos[0], pos[1], pos[2] );
 
		glutSolidSphere(0.075f,10.0f,10.0f);
		glPopMatrix();
}
void SimpleSystem::reset(){
	m_vVecState.clear();
	m_vVecState.push_back(Vector3f(1,0,0));
}