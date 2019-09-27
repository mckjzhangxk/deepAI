#include "MazeHelper.h"
#include "queue.h"
#include "simplepath.h"

void createSimpleMaze(Maze & maze){
    MazeHelper helper(maze);

    pointT current=helper.randomBegin();

    while (!helper.allInclude()) {
        helper.setInclude(current);
        pointT neibour=helper.randomNeighbour(current);
        if(!helper.isInclude(neibour)){
            maze.setWall(current,neibour,false);
        }
        current=neibour;

    }
}

bool mazeSolve(Maze & maze){
    MazeHelper helper(maze);
    Queue<SimplePath> queue;
    queue.enqueue(SimplePath(0,0));

    /*run dfs search to find the target solution*/
    bool found=false;
    Stack<pointT> pp;

    while (!queue.isEmpty()) {
        SimplePath path=queue.dequeue();
        pointT last=path.top();
        if(helper.isFinal(last)){
            found=true;
            pp=path.getPath();
            break;
        }
        Vector<pointT> neibours=helper.neibours(last);
        for(pointT nn:neibours){
            SimplePath newpath=path;
            if(newpath.extendPath(nn))
                queue.enqueue(newpath);
        }
    }
    if(found){
        for(pointT t:pp){
            maze.drawMark(t,"red");
        }
    }
    return found;
}

void play_maze(){
    Maze mz(20,30,true);
    createSimpleMaze(mz);
    mz.draw();
    bool sovable=mazeSolve(mz);
}
