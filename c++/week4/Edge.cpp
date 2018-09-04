//
// Created by zhangxk on 18-9-4.
//

#include "Edge.h"

Edge::Edge(const Node &a, const Node &b, double weight) : a(a), b(b), weight(weight) {}

const Node &Edge::getA() const {
    return a;
}

const Node &Edge::getB() const {
    return b;
}

double Edge::getWeight() const {
    return weight;
}

void Edge::setWeight(double weight) {
    Edge::weight = weight;
}

std::ostream &operator<<(std::ostream &os, const Edge &edge) {
    os << edge.a << "---->" << edge.b << "(weight: " << edge.weight<<"),";
    return os;
}
