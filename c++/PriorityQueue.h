//
// Created by zhangxk on 18-9-4.
//

#ifndef WEEK4_PRIORITYQUEUE_H
#define WEEK4_PRIORITYQUEUE_H

#include <map>
#include <vector>
#include "T.h"
using namespace std;


class PriorityQueue {
public:
    bool empty() {return queue.empty();}
    int size() {return queue.size();}
    int contain(T t);
    void insert(T t) {queue.push_back(t);}
    void update(T t);

    T pop();
private:
    vector <T> queue;
};


#endif //WEEK4_PRIORITYQUEUE_H
