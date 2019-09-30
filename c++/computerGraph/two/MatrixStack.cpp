#include "MatrixStack.h"
#include<iostream>
using namespace std;
MatrixStack::MatrixStack()
{
	
	// Initialize the matrix stack with the identity matrix.
	Matrix4f I=Matrix4f::identity();
	m_matrices.push_back(I);
}

void MatrixStack::clear()
{
	// Revert to just containing the identity matrix.
	m_matrices.clear();
	Matrix4f I=Matrix4f::identity();
	m_matrices.push_back(I);

}

Matrix4f MatrixStack::top()
{
	// Return the top of the stack

	return m_matrices[m_matrices.size()-1];
}

void MatrixStack::push( const Matrix4f& m )
{
	// Push m onto the stack.
	// Your stack should have OpenGL semantics:
	// the new top should be the old top multiplied by m

	
	Matrix4f&  currentState=m_matrices[m_matrices.size()-1];
	/**
	 * I make a bug,multi m at left hand side
	 * 
	 * I need first convert a local to parent using m,
	 * 
	 * then use parent's convert matrix currentState to global
	*/
	m_matrices.push_back(currentState*m);
}

void MatrixStack::pop()
{
	// Remove the top element from the stack
	m_matrices.pop_back();
}
