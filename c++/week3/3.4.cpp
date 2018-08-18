#include <cstring>
#include <iostream>
#include <ctime>
#include <cstdlib>
using namespace std;
//notice c is prefix for c++ lib

/**

	define base struct of Dijkstra Graph,
	define the graph Size, then with 
	prob=0.3 of be a edge,random generate
	a graph

*/
class Graph{

public:
	Graph():SIZE(5){
		srand(clock());
		this->graph=new bool*[SIZE];
		for(int i=0;i<SIZE;i++)
			this->graph[i]=new bool[SIZE];
		//random assign edge between node
		//the expected num of edges is  prob*n*[n-1]/2
		//so as Graph size increase,this is more likely 
		//to connected graph(notice to be connected only
		//need >=n-1 edges)
		for(int i=0;i<SIZE;i++)
			for(int j=i;j<SIZE;j++){
				if(i==j) this->graph[i][j]=false;
				else{
					this->graph[i][j]=this->graph[j][i]=prob();
				}		
			}
			
	}
	~Graph(){
		for(int i=0;i<SIZE;i++)
			delete this->graph[i]; //graph[i] is bool*,a list of bool
		delete graph;
	}
private:
	bool prob(){
		return (rand()%100) <=20;
	}
	const int SIZE;
	bool** graph;

};

int main(int argc,char* argv[]){
	Graph *p_g=new Graph();
	delete p_g;
	return 0;
}
