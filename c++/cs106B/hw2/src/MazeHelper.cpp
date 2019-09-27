#include "MazeHelper.h"


MazeHelper::MazeHelper(Maze & maze):m_mark(Grid<bool>(maze.numRows(),maze.numCols())),m_maze(maze)
{

}
bool MazeHelper::isInclude(const pointT& p){
    if(!m_maze.pointInBounds(p))
        throw "point not in bounds";
    return m_mark.get(p.row,p.col);
}

void MazeHelper::setInclude(const pointT& p){
    if(!m_maze.pointInBounds(p))
        throw "point not in bounds";
    m_mark.set(p.row,p.col,true);
}

bool MazeHelper::allInclude(){
    for(bool x:m_mark){
        if(!x) return false;
    }
    return true;
}
pointT MazeHelper::randomBegin(){
    int r=randomInteger(0,m_maze.numRows()-1);
    int c=randomInteger(0,m_maze.numCols()-1);
    return pointT{r,c};
}

pointT MazeHelper::randomNeighbour(const pointT& p){
    Vector<pointT> vs;

    for(int r=-1;r<=1;r+=2)
    {
            pointT pv{p.row+r,p.col};
            if(m_maze.pointInBounds(pv))
                vs.add(pv);
     }
    for(int c=-1;c<=1;c+=2){
        pointT pv{p.row,p.col+c};
        if(m_maze.pointInBounds(pv))
            vs.add(pv);
    }

    int chioce=randomInteger(0,vs.size()-1);
    return vs.get(chioce);
}

Vector<pointT> MazeHelper::neibours(const pointT &p){
    Vector<pointT> vs;

    for(int r=-1;r<=1;r+=2)
    {
            pointT pv{p.row+r,p.col};
            if(m_maze.pointInBounds(pv)&&!m_maze.isWall(p,pv))
                vs.add(pv);
     }
    for(int c=-1;c<=1;c+=2){
        pointT pv{p.row,p.col+c};
        if(m_maze.pointInBounds(pv)&&!m_maze.isWall(p,pv))
            vs.add(pv);
    }
    return vs;
}
bool MazeHelper::isFinal(const pointT &p){
    return  (p.row==m_maze.numRows()-1)&&(p.col==m_maze.numCols()-1);
}
