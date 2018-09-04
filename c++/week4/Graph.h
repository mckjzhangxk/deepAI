//
// Created by zhangxk on 18-9-4.
//

#ifndef WEEK4_GRAPH_H
#define WEEK4_GRAPH_H
using namespace std;

#include <vector>
#include <ostream>
#include "Node.h"
#include "Edge.h"

class Graph {
public:
    Graph(const int size);
    /*
        returns the number of vertices in the grap
    */
    int V() {return this->vertexes.size();}
    /*
        returns the number of edges in the graph
    */
    int E();
    /*
        tests whether there is an edge from node x to node y.x,y are index of vertex
    */
    bool adjacent(int x,int y);

    vector<Edge>& neighbors(int x);

    void add(int x,int y,double w);

    friend ostream &operator<<(ostream &os, Graph &graph);

    const vector<Node> &getVertexes() const;

    const vector<vector<Edge>> &getEdges() const;
    void montoCarlo(int density);
private:
    std::vector<Node> vertexes;
    std::vector<vector <Edge>> edges;
};


#endif //WEEK4_GRAPH_H
