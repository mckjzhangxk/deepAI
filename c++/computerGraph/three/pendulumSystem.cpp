
#include "pendulumSystem.h"

PendulumSystem::PendulumSystem(int numParticles):ParticleSystem(numParticles),m_gravity(9.8),m_drag(2.5)
{
	reset();
}


// TODO: implement evalF
// for a given state, evaluate f(X,t)
vector<Vector3f> PendulumSystem::evalF(vector<Vector3f> state)
{
	vector<Vector3f> f;
	//fixed point
	f.push_back(Vector3f(0,0,0));
	f.push_back(Vector3f(0,0,0));
	// YOUR CODE HERE
	for (int i = 1; i < m_numParticles+1; i++) {
		vector<float> spring=m_springs[i-1];
		int previous=spring[0];

		Vector3f x0=m_vVecState[2*previous];
		Vector3f x1=m_vVecState[2*i];
		Vector3f v1=m_vVecState[2*i+1];
		
		
		float K=spring[2];
		float restlen=spring[1];

		float L=sqrt((x1-x0).absSquared());
		
		Vector3f string_force=-K*(L-restlen)*(x1-x0).normalized();
		Vector3f drag_force=-m_drag*v1;
		Vector3f gravity=m_gravity*Vector3f(0,-1,0);

		f.push_back(v1);
		f.push_back(string_force+drag_force+gravity);
	}
	return f;
}

// render the system (ie draw the particles)
void PendulumSystem::draw()
{
	for (int i = 0; i < m_numParticles+1; i++) {
		Vector3f pos(m_vVecState[i*2]) ;//  position of particle i. YOUR CODE HERE
		glPushMatrix();
		glTranslatef(pos[0], pos[1], pos[2] );
		glutSolidSphere(0.075f,10.0f,10.0f);
		glPopMatrix();
	}
}
void PendulumSystem::reset(){
	float distance=0.3;

	m_vVecState.clear();
	//fixed point
	m_vVecState.push_back(Vector3f(0,2,0));
	m_vVecState.push_back(Vector3f(0,0,0));

	// fill in code for initializing the state based on the number of particles
	for (int i = 0; i < m_numParticles; i++) {
		
		// for this system, we care about the position and the velocity
		m_vVecState.push_back(Vector3f(0,1.5-i*distance,0));
		m_vVecState.push_back(Vector3f(0,0,0));

		m_springs.push_back({i,0.5,15});
	}
}