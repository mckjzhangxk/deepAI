#ifndef INDEXBUFFER_H
#define INDEXBUFFER_H
#include "commonHeader.h"
#include "myutils.h"


class IndexBuffer
{
public:
    IndexBuffer(unsigned int *indices,unsigned int count);
    void bind() const;
    void ubind() const;
    inline unsigned int count() const{return m_count;};
    ~IndexBuffer();


private:
    unsigned int m_renderID;
    unsigned int m_count;
};

#endif // INDEXBUFFER_H
