//
// Created by zhangxk on 18-9-4.
//

#include "T.h"

T::T(Node &current, Node &parent, double cost) : current(current), parent(parent), cost(cost) {}

Node T::getCurrent() const {
    return current;
}

void T::setCurrent(Node current) {
    T::current = current;
}

Node T::getParent() const {
    return parent;
}

void T::setParent(Node parent) {
    T::parent = parent;
}

double T::getCost() const {
    return cost;
}

void T::setCost(double cost) {
    T::cost = cost;
}

bool T::operator==(const T &rhs) const {
    return current == rhs.current;
}

bool T::operator!=(const T &rhs) const {
    return !(rhs == *this);
}

bool T::operator<(const T &rhs) const {
    return cost < rhs.cost;
}

std::ostream &operator<<(std::ostream &os, const T &t) {
    os << "current: " << t.current << " parent: " << t.parent << " cost: " << t.cost;
    return os;
}


