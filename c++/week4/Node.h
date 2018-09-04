//
// Created by zhangxk on 18-9-4.
//

#ifndef WEEK4_NODE_H
#define WEEK4_NODE_H


#include <ostream>

class Node {

public:
    Node();
    Node(int idx);
    /*
     *
     * compare between idx
     * */
    bool operator==(const Node &rhs) const;

    friend std::ostream &operator<<(std::ostream &os, const Node &node);

    bool operator!=(const Node &rhs) const;

    int getIdx() const;

    void setIdx(int idx);


private:
    int idx;

};


#endif //WEEK4_NODE_H
