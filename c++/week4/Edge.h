//
// Created by zhangxk on 18-9-4.
//

#ifndef WEEK4_EDGE_H
#define WEEK4_EDGE_H


#include <ostream>
#include "Node.h"

class Edge {
public:
    Edge(const Node &a, const Node &b, double weight);

    const Node &getA() const;

    const Node &getB() const;

    double getWeight() const;

    void setWeight(double weight);

    friend std::ostream &operator<<(std::ostream &os, const Edge &edge);

private:
    const Node& a;
    const Node& b;
    double weight;
};


#endif //WEEK4_EDGE_H
