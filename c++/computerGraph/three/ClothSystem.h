#ifndef CLOTHSYSTEM_H
#define CLOTHSYSTEM_H

#include <vecmath.h>
#include <vector>
#include<math.h>
#include "particleSystem.h"
#include <GL/glut.h>
#include <iostream>

struct Spring
{
	int from;
	int to;
	float k;
	float r0;
};

class ClothSystem: public ParticleSystem
{
///ADD MORE FUNCTION AND FIELDS HERE
public:
	ClothSystem();
	vector<Vector3f> evalF(vector<Vector3f> state);
	void toggleWired();
	void draw();
	void reset();
private:
	int indexOf(int i,int j) const;
	void buildGridPaticle();
	void buildSprings();

	int mrows;
	int mcols;
	const float CONST_CLOTH_WIDTH;
	const float CONST_CLOTH_HEIGHT;
	const float CONST_GRAVITY;
	const float CONST_DRAG;

	bool showWired;
	vector<Spring> mSprings;
};


#endif
