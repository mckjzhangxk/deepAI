#ifndef SIMPLEPATH_H
#define SIMPLEPATH_H
#include "set.h"
#include "stack.h"
#include "maze.h"

class SimplePath
{
public:
    SimplePath();
    SimplePath(int r,int c);

    const Stack<pointT> & getPath();
    /*
     * add p in current if p have not occur in path
     * otherwise retrn false
     *
    */
    bool extendPath(const pointT &p);
    /*
     * get final node of path
    */
    const pointT & top();
    int size();

private:
    Stack<pointT> m_path;
    Set<pointT> m_nodes;
};

#endif // SIMPLEPATH_H
