#include "simplepath.h"
int comparePoint(const pointT& a,const pointT& b){
    if(a.row<b.row) return -1;
    else if(a.row==b.row){
        if(a.col<b.col) return -1;
        else if(a.col==b.col) return 0;
        else return 1;
    }else
        return 1;
}
SimplePath::SimplePath():m_nodes(comparePoint)
{

}
SimplePath::SimplePath(int r,int c):m_nodes(comparePoint)
{
    pointT p{r,c};
    m_nodes.add(p);
    m_path.push(p);
}

const Stack<pointT> &SimplePath::getPath(){
    return m_path;
}

bool SimplePath::extendPath(const pointT &p){
    if(m_nodes.contains(p)){
        return false;
    }
    m_nodes.add(p);
    m_path.push(p);
    return true;
}

const pointT& SimplePath::top(){
    return m_path.top();
}
int SimplePath::size(){
    return m_path.size();
}
