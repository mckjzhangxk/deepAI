#ifndef VERTEXLAYOUT_H
#define VERTEXLAYOUT_H
#include "commonHeader.h"
#include <vector>
using namespace  std;

struct LayoutElement{
    GLenum type;
    unsigned int count;
    GLboolean normalized;
    unsigned int offset;

    static int compute_size(unsigned int x){
        switch (x) {
            case GL_FLOAT:return sizeof (GLfloat);
            case GL_UNSIGNED_INT:return sizeof (GLuint);
            case GL_INT:return sizeof (GLint);
        }
    }
};
class VertexLayout
{
public:
    VertexLayout();
//    template<typename T>
    void add(unsigned int count);
    inline vector<LayoutElement> & getLayout(){return  m_layout;};
    inline unsigned int get_stride(){return m_stride;};
private:
    vector<LayoutElement> m_layout;
    unsigned int m_stride;
};

#endif // VERTEXLAYOUT_H
