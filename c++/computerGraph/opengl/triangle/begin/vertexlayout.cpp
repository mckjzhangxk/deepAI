#include "vertexlayout.h"

VertexLayout::VertexLayout():m_stride(0)
{

}
//template<typename  T>
void VertexLayout::add(unsigned int count)
{
    LayoutElement layout={GL_FLOAT,count,false,m_stride};
    m_stride+=count*sizeof (GLfloat);
    m_layout.push_back(layout);
}
