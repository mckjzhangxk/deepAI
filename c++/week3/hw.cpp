#include <cstdlib>
#include <iostream>
#include <vector>
#include <list>
#include <ctime>
#include <map>

using namespace std;
/*
tuple is used to store adjcent node

*/
class tuple{
public:
	tuple(int idx,double dis){
		this->idx=idx;
		this->distance=dis;
	}	
	int getIdx(){
		return this->idx;
	}
	double getDistance(){
		return this->distance;
	}
private:
	int idx;
	double distance;
};

class Graph{
public:
	Graph(int size=50){
		this->size=size;
		/*
			nodes are store in verctors,so I can
		access every node as O(1)
		*/
		this->vertices=vector<list<tuple> >(size);
	}

	int V(){
		
	/*
	  returns the number of vertices in the grap
	*/
		return this->size;
	}
	int E(){
	/*
		returns the number of edges in the graph
	*/
		int count=0;
		for(int i=0;i<this->vertices.size();i++){
			count=count+this->vertices[i].size();
		}
	
	}

	bool adjacent(int x,int y){
		/*
		 tests whether there is an edge from node x to node y.
		x,y are index of vertex
		*/
		list<tuple> xadj=this->vertices[x];
	
		for(list<tuple>::iterator it=xadj.begin();it!=xadj.end();it++){
			int t=(*it).getIdx();
			if(t==y) return true;
		}
		return false;
	}
	vector<int> neighbors(int x){
		/*
		 lists all nodes y such that there is an edge from x to y.
		*/
		vector<int> ret=vector<int>();
		list<tuple> xadj=this->vertices[x];
		for(list<tuple>::iterator it=xadj.begin();it!=xadj.end();it++){
                        int t=(*it).getIdx();
                        ret.push_back(t);
                }
		return ret;
	}
	void add(int x,int y,double w=0){
		/*
		adds to G the edge from x to y, if it is not there.
		*/
		if(this->adjacent(x,y)==false){
			list<tuple>* xadj=&this->vertices[x];
			xadj->push_back(tuple(y,w));
		}
	}
	friend ostream & operator<<(ostream & out,Graph& g){
		for(int i=0;i<g.size;i++){
			out<<i<<":";
			list<tuple> adj=g.vertices[i];
			for(list<tuple>::iterator it=adj.begin();it!=adj.end();it++){
				out<<(*it).getIdx()<<"("<<(*it).getDistance()<<"),";
			}
			out<<endl;
		}
		return out;
	}
	void montoCarlo(int density){
		srand(clock());
		for(int i=0;i<this->size;i++)
			for(int j=i+1;j<this->size;j++){
				if((rand()%100)<=density){
					int w=rand()%30;
					this->add(i,j,w);
					this->add(j,i,w);
					
				}	
			}
	}	
private:
	int size;
	vector<list<tuple> > vertices;
};
class PriorityQueue{
public:
	 PriorityQueue(){
	}
	void insert(int node,tuple* a){
		this->queue[node]=a;
	}
	bool contain(int node){
		map<int,tuple*>::iterator it=queue.find(node);
		return it!=queue.end();
	}
	int top(){
		int min=99999;
		int ret=-1;
		for(map<int,tuple*>::iterator it=queue.begin();it!=queue.end();it++){
			tuple * element=it->second;
			if(element->getDistance()<min){
				min=element->getDistance();
				ret=0;
			}
		}
		return ret;
	}
	bool empty(){
		return this->queue.empty();
	}
	int size(){
		return this->queue.size();
	}
private:
	map<int,tuple*> queue;
};

int main(int argc,char * argv[]){
	Graph g=Graph(10);
	g.montoCarlo(30);
	cout<<g<<endl;
	return 0;
}
