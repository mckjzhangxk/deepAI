//
// Created by zhangxk on 18-9-4.
//

#ifndef WEEK4_T_H
#define WEEK4_T_H


#include <ostream>
#include "Node.h"

class T {
public:
    T(Node &current, Node &parent, double cost);

    Node getCurrent() const;

    void setCurrent(Node current);

    Node getParent() const;

    void setParent(Node parent);

    double getCost() const;

    void setCost(double cost);

    bool operator<(const T &rhs) const;

    bool operator==(const T &rhs) const;

    bool operator!=(const T &rhs) const;

    friend std::ostream &operator<<(std::ostream &os, const T &t);

private:
    Node current;
    Node parent;
    double cost;
};


#endif //WEEK4_T_H
