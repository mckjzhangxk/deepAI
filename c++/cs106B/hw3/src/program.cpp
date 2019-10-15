#include "vector.h"
#include "set.h"
#include "iostream"
#include "gwindow.h"
using namespace std;

int CountWays(int numStairs,Vector<char> step_sofar){
    if(numStairs==0){
        for(char st:step_sofar)
            cout<<st<<",";
        cout<<endl;
        return 1;
    }
    int ret=0;
    if(numStairs>=2){
        Vector<char> nstep_sofar=step_sofar;
        nstep_sofar.add('L');
        ret+=CountWays(numStairs-2,nstep_sofar);
    }
    if(numStairs>=1){
        Vector<char> nstep_sofar=step_sofar;
        nstep_sofar.add('S');
        ret+=CountWays(numStairs-1,nstep_sofar);
    }
    return ret;
}
int CountWays(int numStairs){
    return CountWays(numStairs,Vector<char>());
}

void DrawRuler(double x,double y,double w, double h,GWindow * window){
    if(h<20)
        return;

    window->setColor("black");
    window->setLineWidth(3);
    window->drawLine(x,y,x+w,y);
    window->drawLine(x+w/2,y,x+w/2,y-h/2);

    DrawRuler(x,y,w/2,h/2,window);
    DrawRuler(x+w/2,y,w/2,h/2,window);
}

void DrawRuler(double x,double y,double w, double h){
    GWindow * window=new GWindow(1024,768);

    window->center();

    DrawRuler(x,y,w,h,window);

}


/*3.Every Count Problem
    define 2 global variable to store total votes,and target vote
*/
static int TOTAL_VOTE;
static int TARGET_VOTE;
bool isCriticalVote(Vector<int>& v){
    int voteMe=0;
    for(int n:v){
        voteMe+=n;
    }
    int voteOthers=TOTAL_VOTE-voteMe;

    return (voteMe>voteOthers && voteMe<voteOthers+TARGET_VOTE) ||(voteMe<voteOthers && voteMe+TARGET_VOTE>voteOthers);
}
//int CountCriticalVotes(Vector<int> blocks,Vector<int> currentVote){
//    if(blocks.isEmpty())
//        return 0;

//    bool isCritical=isCriticalVote(currentVote);


//    int nextVote=blocks.pop_front();
//    Vector<int>voteMe(currentVote);
//    voteMe.add(nextVote);

//    int voteMeCount=CountCriticalVotes(blocks,voteMe);
//    int notVoteMeCount=CountCriticalVotes(blocks,currentVote);

//    return isCritical+voteMeCount+notVoteMeCount;
//}
int CountCriticalVotes(Vector<int> blocks,Vector<int> currentVote){
    if(blocks.isEmpty()){
        return isCriticalVote(currentVote);
    }
    int nextVote=blocks.pop_front();
    Vector<int>voteMe(currentVote);
    voteMe.add(nextVote);

    int voteMeCount=CountCriticalVotes(blocks,voteMe);
    int notVoteMeCount=CountCriticalVotes(blocks,currentVote);

    return voteMeCount+notVoteMeCount;
}
int CountCriticalVotes(Vector<int> &blocks,int blockIndex){
    TARGET_VOTE=blocks[blockIndex];
    blocks.remove(blockIndex);


    int sum=0;
    for(int n:blocks){
        sum+=n;
    }
    TOTAL_VOTE=sum;
    return CountCriticalVotes(blocks,Vector<int>());
}
int CountCriticalVotes(Vector<int> &&blocks,int blockIndex){
    return CountCriticalVotes(blocks,blockIndex);
}


/*5.recursive puzzle

*/
bool solable(int start,Vector<int> & square,Set<int >&visited){

    if(square.get(start)==0)
        return true;
    else if(visited.contains(start))
        return false;
    visited.add(start);

    int left=start-square.get(start);
    if(left>=0&&solable(left,square,visited)){
        cout<<"L ";
        return true;
    }

    int right=start+square.get(start);
    if(right<square.size()&&solable(right,square,visited)){
        cout<<"R ";
        return true;
    }
    return  false;
}
bool solable(int start,Vector<int> & square){
    Set<int>visited;
    bool r=solable(start,square,visited);
    cout<<endl;
    return r;
}
bool solable(int start,Vector<int> && square){
    return solable(start,square);

}
