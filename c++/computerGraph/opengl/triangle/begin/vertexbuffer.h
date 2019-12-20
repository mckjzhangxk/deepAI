#ifndef VERTEXBUFFER_H
#define VERTEXBUFFER_H
#include "commonHeader.h"
#include "myutils.h"

class VertexBuffer
{
public:
    VertexBuffer(GLfloat *data,unsigned int size);
    void bind() const;
    void ubind() const;
    ~VertexBuffer();
private:
    unsigned int m_renderID;
};

#endif // VERTEXBUFFER_H
