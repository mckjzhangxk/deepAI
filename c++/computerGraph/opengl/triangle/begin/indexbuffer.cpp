#include "indexbuffer.h"
IndexBuffer::IndexBuffer(unsigned int *indices, unsigned int count)
{
    ASSERT(glGenBuffers(1,&m_renderID));
    ASSERT(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_renderID));
    ASSERT(glBufferData(GL_ELEMENT_ARRAY_BUFFER,count*sizeof(unsigned int),indices,GL_STATIC_DRAW));
}

void IndexBuffer::bind() const
{
    ASSERT(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_renderID));
}

void IndexBuffer::ubind() const
{
    ASSERT(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0));
}

IndexBuffer::~IndexBuffer()
{
    ASSERT(glDeleteBuffers(1,&m_renderID));
}
