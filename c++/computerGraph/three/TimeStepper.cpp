#include "TimeStepper.hpp"
#include<iostream>
///TODO: implement Explicit Euler time integrator here
void ForwardEuler::takeStep(ParticleSystem* particleSystem, float stepSize)
{
    vector<Vector3f> state=particleSystem->getState();
    vector<Vector3f> dstate=particleSystem->evalF(state);

    for(unsigned i=0;i<state.size();i++){
        state[i]+=stepSize*dstate[i];
    }

    particleSystem->setState(state);
}

///TODO: implement Trapzoidal rule here
void Trapzoidal::takeStep(ParticleSystem* particleSystem, float stepSize)
{
    vector<Vector3f> state=particleSystem->getState();

    vector <Vector3f> d1=particleSystem->evalF(state);
    
    vector <Vector3f> fakeState;
    for(unsigned i=0;i<state.size();i++){
        fakeState.push_back(state[i]+stepSize*d1[i]);
    }

    vector<Vector3f> d2=particleSystem->evalF(fakeState);


    for(unsigned i=0;i<state.size();i++){
        state[i]+=0.5*stepSize*(d1[i]+d2[i]);
    }
    particleSystem->setState(state);
}
