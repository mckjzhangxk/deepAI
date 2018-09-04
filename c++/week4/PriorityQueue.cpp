//
// Created by zhangxk on 18-9-4.
//

#include "PriorityQueue.h"

int PriorityQueue::contain(T &node) {
    for (int i = 0; i <queue.size() ; ++i) {
        if(node==queue[i])
            return i;
    }
    return -1;
}


void PriorityQueue::update(T t) {
    int index=contain(t);
    if(index>-1 &&t<queue[index]){
        queue[index].setCost(t.getCost());
        queue[index].setParent(t.getParent());
    } else{
        insert(t);
    }
}

T PriorityQueue::pop() {
    int idx=0;
    for (int i = 1; i <queue.size() ; ++i) {
        if(queue[i]<queue[idx]){
            idx=i;
        }
    }
    vector<T>::iterator a=queue.begin()+idx;

    queue.erase(a);
    return queue[idx];
}
