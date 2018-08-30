#include <cstdlib>
#include <iostream>
#include <vector>
#include <list>
using namespace std;

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
		/*for(int i=0;i<xadj.size();i++){
	 		int t=xadj[i].getIdx();
			if(t==i) return true;	
		}*/
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
			list<tuple> xadj=this->vertices[x];
			xadj.push_back(tuple(y,w));
		}
	}

	
private:
	int size;
	vector<list<tuple> > vertices;
};

int main(int argc,char * argv[]){
	Graph g=Graph(10);
	cout<<g.adjacent(0,1)<<endl; 
	Graph* pg=new Graph(12);
	return 0;
}
