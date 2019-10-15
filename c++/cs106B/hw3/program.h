#ifndef PROGRAM_H
#define PROGRAM_H
int CountWays(int numStairs);
void DrawRuler(double x,double y,double w, double h);
int CountCriticalVotes(Vector<int> &blocks,int blockIndex);
int CountCriticalVotes(Vector<int> &&blocks,int blockIndex);

bool solable(int start,Vector<int> & square);
bool solable(int start,Vector<int> && square);
#endif // PROGRAM_H
