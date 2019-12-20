#include "vertexbuffer.h"

VertexBuffer::VertexBuffer(GLfloat *data,unsigned int size)
{
    ASSERT(glGenBuffers(1,&m_renderID));
    ASSERT(glBindBuffer(GL_ARRAY_BUFFER,m_renderID));
    ASSERT(glBufferData(GL_ARRAY_BUFFER,size,data,GL_STATIC_DRAW));
}

void VertexBuffer::bind() const
{
    ASSERT(glBindBuffer(GL_ARRAY_BUFFER,m_renderID));
}

void VertexBuffer::ubind() const
{
    ASSERT(glBindBuffer(GL_ARRAY_BUFFER,0));
}

VertexBuffer::~VertexBuffer()
{
    ASSERT(glDeleteBuffers(1,&m_renderID));
}
