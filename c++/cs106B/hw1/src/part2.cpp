#include "simpio.h"
#include "random.h"
#include "strlib.h"
#include <iostream>
using namespace  std;
/*
 * total N voter,0.5+p vote A,0.5-p vote B
 * when vote, have err probibility to make a
 * error,return whether the final result did not
 * change
 *
*/
bool one_try(int N,double p,double err){
    int numA=N*(0.5+p);
    int numB=N-numA;

    int voteA=0;
    int voteB=0;
    for(unsigned i=0;i<numA;i++){
        if(randomReal(0,1)>err)
            voteA++;
        else
            voteB++;
    }

    for(unsigned i=0;i<numB;i++){
        if(randomReal(0,1)>err)
            voteB++;
        else
            voteA++;
    }
    return ((numA>numB)&&(voteA>voteB)) ||((numA<numB)&&(voteA<voteB));
}
void simulate(int trails=500){
    string voters=getLine("Enter number of voters");
    string diff=getLine("Enter percentage spread between candidates:");
    string errrate=getLine("Enter voting error percentage:");

    int N=stringToReal(voters);
    double percent=stringToReal(diff);
    double err=stringToReal(errrate);

    int invalid=0;
    for(unsigned i=0;i<trails;i++){
        if(!one_try(N,percent,err))
            invalid++;
    }

    cout<<"Chance of an invalid election result after "<<trails<<" traits= "<<(100*double(invalid)/trails)<<"%";
}
