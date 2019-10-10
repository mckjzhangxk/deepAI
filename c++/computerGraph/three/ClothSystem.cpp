#include "ClothSystem.h"

//TODO: Initialize here
ClothSystem::ClothSystem():
	CONST_CLOTH_WIDTH(1.6f),CONST_CLOTH_HEIGHT(1.6),CONST_GRAVITY(3),CONST_DRAG(10.0),
	mrows(8),mcols(8),showWired(false)
{
	reset();
}


// TODO: implement evalF
// for a given state, evaluate f(X,t)
vector<Vector3f> ClothSystem::evalF(vector<Vector3f> state)
{
	vector<Vector3f> f;
	
	//first eval force only relay on particle itself
	for(int i=0;i<state.size();i++){
		Vector3f x=state[2*i];
		Vector3f v=state[2*i+1];

		Vector3f g_force=-CONST_GRAVITY*Vector3f(0,1,0);
		Vector3f d_force=-CONST_DRAG*v;

		f.push_back(v);
		f.push_back(g_force+d_force);
		// f.push_back(Vector3f());
		// f.push_back(Vector3f());
	}
	//then deal with string
	for(Spring sf:mSprings){
		int A=sf.from;
		int B=sf.to;
		float K=sf.k;
		float r0=sf.r0;
		Vector3f xA=m_vVecState[2*A];
		Vector3f xB=m_vVecState[2*B];
		
		float L=sqrt((xA-xB).absSquared());

		Vector3f fA=-K*(L-r0)*(xA-xB).normalized();
		Vector3f fB=-fA;

		
		if(A==0){
			cout<<A<<"-->"<<B<<endl;
			cout<<"L:"<<L<<endl;
			cout<<"force:"<<K*(L-r0)<<endl;
			cout<<"----------"<<endl;
		}
		f[2*A+1]+=fA;
		f[2*B+1]+=fB;
	}
	f[2*indexOf(0,0)]=Vector3f(0,0,0);
	f[2*indexOf(0,0)+1]=Vector3f(0,0,0);

	f[2*indexOf(0,mcols-1)]=Vector3f(0,0,0);
	f[2*indexOf(0,mcols-1)+1]=Vector3f(0,0,0);

	return f;
}
void ClothSystem::toggleWired(){
	showWired=!showWired;
}
///TODO: render the system (ie draw the particles)
void ClothSystem::draw()
{
	// for (int i = 0; i < m_vVecState.size(); i++) {
	// 	Vector3f pos(m_vVecState[i*2]) ;//  position of particle i. YOUR CODE HERE
	// 	glPushMatrix();

	// 	glBegin(GL_LINE_STRIP);
	// 	glEnd();
	// 	glTranslatef(pos[0], pos[1], pos[2] );
	// 	glutSolidSphere(0.075f,10.0f,10.0f);
	// 	glPopMatrix();
	// }

	for(int r=0;r<mrows-1;r++)
		for(int c=0;c<mcols-1;c++){
			Vector3f v1=m_vVecState[2*indexOf(r,c)];
			Vector3f v2=m_vVecState[2*indexOf(r+1,c)];
			Vector3f v3=m_vVecState[2*indexOf(r+1,c+1)];
			Vector3f v4=m_vVecState[2*indexOf(r,c+1)];
			


			if(showWired){
				glBegin(GL_LINE_LOOP);
			}else{
				glBegin(GL_POLYGON);
			}
				glVertex3f(v1.x(),v1.y(),v1.z());
				glVertex3f(v2.x(),v2.y(),v2.z());
				glVertex3f(v3.x(),v3.y(),v3.z());
				glVertex3f(v4.x(),v4.y(),v4.z());
			glEnd();
		}
}

int ClothSystem::indexOf(int i,int j) const{
	return i*mcols+j;
}
//init mSprings,using indexof functoin
void ClothSystem::buildSprings(){
	
	float K0=50;
	float R0=0.2;

	float K1=50;
	float R1=0.2*sqrt(2);

	float K2=50;
	float R2=2*0.2;

	for(int r=0;r<mrows;r++){
		for(int c=0;c<mcols;c++){
			//grid springs
			if(r+1<mrows)
				mSprings.push_back({indexOf(r,c),indexOf(r+1,c),K0,R0});
			if(c+1<mcols)
				mSprings.push_back({indexOf(r,c),indexOf(r,c+1),K0,R0});
			
			//shear springs
			if(r+1<mrows&&c+1<mcols)
				mSprings.push_back({indexOf(r,c),indexOf(r+1,c+1),K1,R1});
			if(r+1<mrows&&c-1>=0)
				mSprings.push_back({indexOf(r,c),indexOf(r+1,c-1),K1,R1});
			//flex spring
			if(r+2<mrows)
				mSprings.push_back({indexOf(r,c),indexOf(r+2,c),K2,R2});
			if(c+2<mcols)
				mSprings.push_back({indexOf(r,c),indexOf(r,c+2),K2,R2});
		}
	}
}
//initial m_vVecState base on mrows,mcols,CLOTH_WIDTH,CLOTH_HEIGHT
void ClothSystem::buildGridPaticle(){
	float cellwidth=CONST_CLOTH_WIDTH/(mcols);
	float cellheight=CONST_CLOTH_HEIGHT/(mrows);

	for(int r=0;r<mrows;r++){
		for(int c=0;c<mcols;c++){
			Vector3f x(cellwidth*c,3,cellheight*r);
			Vector3f v(0,0,0);
			m_vVecState.push_back(x);
			m_vVecState.push_back(v);
		}
	}
}
void ClothSystem::reset(){
	m_vVecState.clear();
	buildGridPaticle();
	mSprings.clear();
	buildSprings();
}
