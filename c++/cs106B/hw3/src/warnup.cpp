#include "vector.h"

bool sumEqual(Vector<int>& num,int target){
    int sum=0;
    for(int n:num){
        sum+=n;
    }
    return sum==target;
}
bool canMakeSum(Vector<int> sofar,Vector<int> rest,int target){
    if(sumEqual(sofar,target)){
        return true;
    }
    if(rest.isEmpty()){
        return false;
    }
    if(canMakeSum(sofar,rest.subList(1),target)){
        return true;
    }else{
        sofar.add(rest.pop_front());
        if(canMakeSum(sofar,rest,target))
            return true;
    }
    return false;
}
bool canMakeSum(Vector<int> &num,int target){
    Vector<int> sofar;
    return canMakeSum(sofar,num,target);

}
