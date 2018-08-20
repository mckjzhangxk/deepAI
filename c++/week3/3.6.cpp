#include <cstring>
#include <iostream>
#include <ctime>
#include <cstdlib>
using namespace std;
//notice c is prefix for c++ lib

/**
	continue on class 3.4,define a is_connect function

	define base struct of Dijkstra Graph,
	define the graph Size, then with 
	prob=0.3 of be a edge,random generate
	a graph

*/
class Graph{

public:
	Graph():SIZE(30){
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
	/**
		the process is follow:
		1.openset[],closeset[] with size Graph.Size,initial to false,initial openset[0] to 1
		repeat size times(because one iteration add 1 from open to close):
		   score old_size=c_size
		   if 
			find a node in open set but not in close set(open[i] ==1 and close[i]==0)
			add it to closeset,and,add all it's adjcent to openset
		   else 
			go to next node
		   
		   after one pass throgh all nodes,if c_size don't change ,terminate with false
	**/
	bool is_connect(){
		bool* open=new bool[SIZE];
		bool* close=new bool[SIZE];

		//initial to 0,except open[0]=1
		for(int i=0;i<SIZE;i++){
			open[i]=close[i]=false;
		}
		open[0]=true;
		
		//iteration,at most SIZE step,one step move node from openset to close
		int c_size=0;//num of node in closeset
		for(int k=0;k<SIZE;k++){
			int c_old_size=c_size;
			for(int i=0;i<SIZE;i++){
				if(open[i]&&!close[i]){
					c_size++;
					close[i]=1; //move i fron open to close
					//update openset
					for(int j=0;j<SIZE;j++)
						open[j]=open[j]||this->graph[i][j];
					
				}
			}
			//after one iteration
			if(c_size==c_old_size)
				return false;
			if(c_size==SIZE)
				return true;
		}
		return c_size==SIZE; 
	}

	~Graph(){
		for(int i=0;i<SIZE;i++)
			delete this->graph[i]; //graph[i] is bool*,a list of bool
		delete graph;
	}
private:
	bool prob(){
		return (rand()%100) <=10;
	}
	const int SIZE;
	bool** graph;

};

int main(int argc,char* argv[]){
	Graph *p_g=new Graph();
	cout<<p_g->is_connect()<<endl;
	delete p_g;
	return 0;
}
