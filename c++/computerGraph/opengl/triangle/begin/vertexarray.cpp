#include "vertexarray.h"

void VertexArray::bind()
{
    ASSERT(glBindVertexArray(m_renderID));
}

void VertexArray::unbind()
{
    ASSERT(glBindVertexArray(0));
}

VertexArray::VertexArray(VertexBuffer &vb,VertexLayout & vlayout)
{
    ASSERT(glGenVertexArrays(1,&m_renderID));
    ASSERT(glBindVertexArray(m_renderID));

    vb.bind();
    vector<LayoutElement>& layout=vlayout.getLayout();

    for (int i=0;i<layout.size();i++){
        glVertexAttribPointer(i,layout[i].count,layout[i].type,
                              layout[i].normalized,vlayout.get_stride(),
                              (const GLvoid *)layout[i].offset);
        glEnableVertexAttribArray(i);
    }

}

VertexArray::~VertexArray()
{
    ASSERT(glDeleteVertexArrays(1,&m_renderID));
}
