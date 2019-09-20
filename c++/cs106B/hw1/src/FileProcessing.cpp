#include"vector.h"
#include<iostream>
#include<fstream>
#include "tokenscanner.h"
#include"simpio.h"
#include "strlib.h"

using namespace  std;


void getInput( Vector<double>& scores){

    string filename;
    ifstream fin;
    while (true) {
        filename=getLine("Input a file name:");
        fin=ifstream(filename);
        if(fin) break;
        cout<<"invalid file name!"<<endl;
    }

    TokenScanner sc(fin);
    sc.ignoreWhitespace();
    while (sc.hasMoreTokens()) {
        double score=stringToReal(sc.nextToken());
        scores.add(score);
    }
}
void showHistograph(Vector<double>& scores){
    int level[10]={0};
    for(int i=0;i<scores.size();i++){
        int index=scores[i]/10;
        level[index]++;
    }

    for(int i=0;i<10;i++){
        cout<<i*10<<"-"<<(i+1)*10-1<<":";
        int loops=level[i];
        for(int j=0;j<loops;j++)
            cout<<"x";
       cout<<endl;
    }
}

void part4(){
     Vector<double> scores;
     getInput(scores);
     showHistograph(scores);
}
