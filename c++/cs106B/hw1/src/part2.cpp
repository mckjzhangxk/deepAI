#include "simpio.h"
#include "random.h"
#include "strlib.h"
#include <iostream>
using namespace  std;
void simulate(){
    string voters=getLine("Enter number of voters");
    string diff=getLine("Enter percentage spread between candidates:");
    string errrate=getLine("Enter voting error percentage:");
    std::cout<<randomReal(0,1)<<std::endl;
}
