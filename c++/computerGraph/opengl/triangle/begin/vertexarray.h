#ifndef VERTEXARRAY_H
#define VERTEXARRAY_H
#include "vertexbuffer.h"
#include "vertexlayout.h"


class VertexArray
{
public:
    void bind();
    void unbind();
    VertexArray(VertexBuffer &vb,VertexLayout & layout);
    ~VertexArray();
private:
    unsigned int m_renderID;
};

#endif // VERTEXARRAY_H
