//
// Created by zhangxk on 18-9-4.
//

#include "Node.h"

Node::Node(int idx) : idx(idx){}

int Node::getIdx() const {
    return idx;
}

void Node::setIdx(int idx) {
    Node::idx = idx;
}



std::ostream &operator<<(std::ostream &os, const Node &node) {
    os<< node.idx;
    return os;
}

bool Node::operator==(const Node &rhs) const {
    return idx == rhs.idx;
}

bool Node::operator!=(const Node &rhs) const {
    return !(rhs.idx == idx);
}

Node::Node() {

}
