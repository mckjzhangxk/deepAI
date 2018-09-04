//
// Created by zhangxk on 18-9-4.
//

#include "Graph.h"


Graph::Graph(const int size) {
    this->vertexes=vector<Node>(size);
    this->edges=vector<vector <Edge>>(size);
}

int Graph::E() {
    int count=0;
    for(int i=0;i<this->edges.size();i++){
        count+=this->edges[i].size();
    }
    return count;
}

bool Graph::adjacent(int x, int y) {
    vector<Edge> adjcentX=edges[x];

    for(int i=0;i<adjcentX.size();i++){
        if(adjcentX[i].getB()==y)
            return true;
    }
    return false;
}

vector <Edge>& Graph::neighbors(int x) {
    return this->edges[x];
}

void Graph::add(int x, int y, double w) {
    Node& nodex=vertexes[x];
    Node& nodey=vertexes[y];

    nodex.setIdx(x);
    nodey.setIdx(y);
    if(!adjacent(x,y)){
        vector<Edge>& adjX=edges[x];
        adjX.push_back(Edge(nodex,nodey,w));
    }

}

const vector <Node> &Graph::getVertexes() const {
    return vertexes;
}

const vector <vector<Edge>> &Graph::getEdges() const {
    return edges;
}

void Graph::montoCarlo(int density) {
    srand(clock());

    int size=vertexes.size();

    for(int i=0;i<size;i++)
        for(int j=i+1;j<size;j++){
            if((rand()%100)<=density){
                int w=rand()%30;
                this->add(i,j,w);
                this->add(j,i,w);

            }
        }
}

ostream &operator<<(ostream &os, Graph &graph) {
    for (int i = 0; i < graph.V(); ++i) {
        os<<graph.getVertexes()[i]<<":";
        vector<Edge> adj=graph.getEdges()[i];
        for (int j = 0; j <adj.size() ; ++j) {
            os<<adj[j];
        }
        os<<endl;
    }
    return os;
}
