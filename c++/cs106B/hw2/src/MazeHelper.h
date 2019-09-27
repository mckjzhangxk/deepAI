#ifndef MAZECREATOR_H
#define MAZECREATOR_H
#include "maze.h"
#include "vector"

class MazeHelper
{
public:
    MazeHelper(Maze & maze);
    /*
     *
     *check p in in bound,if not,throw a error,
     * then return whether
     * p in marked include
    */
    bool isInclude(const pointT& p);
    /*
     *check p in bound,throw a error if not,
     * mark p is be included
    */
    void setInclude(const pointT& p);
    /*
     * random chioce a start point
    */
    /*
     * tell your all cell is marked include
    */
    bool allInclude();
    pointT randomBegin();
    /*
     * random chioce a neibour of p
    */
    pointT randomNeighbour(const pointT& p);

    /*
     * return all neibours of p
    */
    Vector<pointT> neibours(const pointT& p);
    /*
     * return true if p as upper right conner
    */
    bool isFinal(const pointT&p);
private:
    //true mean include,false mean exclude
    Grid<bool> m_mark;
    Maze&  m_maze;
};

#endif // MAZECREATOR_H
