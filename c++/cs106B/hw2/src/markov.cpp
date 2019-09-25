#include<iostream>
#include<fstream>
#include"simpio.h"
#include "tokenscanner.h"
#include "map.h"
#include "vector.h"
#include "random.h"

using namespace  std;
string readFile(){
    ifstream ifin;
    while (true) {
        string filename=getLine("input a valid file name:");

        ifin.open(filename);
        if(ifin.fail()){
            ifin.clear();
            cout<<"Invalid file!"<<endl;
        }else break;
    }
    //need to change so that can read more chars
    unsigned int MAXBUFFER=1024*1024;
    char buf[MAXBUFFER];
    ifin.read(buf,MAXBUFFER);
    string ret=buf;

    return ret;
}
void updateMap(Map<string,Map<char,int> > &m,string prefix,char suffix){
    if(!m.containsKey(prefix)){
        m.put(prefix,Map<char,int>());
    }
    Map<char,int>& stats=m[prefix];
    if(stats.containsKey(suffix)){
        stats[suffix]+=1;
    }else{
        stats[suffix]=1;
    }
}
string initString(Map<string,Map<char,int> > &m){
    Vector<string> keys=m.keys();
    int maxfreq=-1;
    string mostFrequencePrefix="";

    for(int i=0;i<keys.size();i++){
        Map<char,int> st=m[keys[i]];
        int freq=0;
        for(int v:st.values()){
            freq+=v;
        }
        if(freq>maxfreq){
            maxfreq=freq;
            mostFrequencePrefix=keys[i];
        }
    }
    return mostFrequencePrefix;
}
char chioce_next(Map<string,Map<char,int> > &m,string key){
    if(!m.containsKey(key))
        return 0;
    Map<char,int> candidates=m[key];
    Vector<char> chars;
    Vector<int> fs;
    int Freqs=0;
    for(char c:candidates.keys()){
        chars.add(c);
        fs.add(candidates[c]);
        Freqs+=candidates[c];
    }
    int chioce=randomInteger(1,Freqs);
    for(int i=0;i<fs.size();i++){
        if(chioce>fs[i])
            chioce-=fs[i];
        else
            return chars[i];
    }
    return 0;
}
string random_parse(Map<string,Map<char,int> > &m,int maxchars=2000){
    string initseed=initString(m);
    string prefix=initseed;
    for(int i=0;i<maxchars;i++){
        char nextchar=chioce_next(m,prefix);
        if(!nextchar)
            break;
        initseed.push_back(nextchar);
        prefix=prefix.substr(1)+nextchar;
    }
    return initseed;
}
void markov_main(){
    string s=readFile();
    int K=getInteger("input order:");
    Map<string,Map<char,int> > dict;

    string seed=s.substr(0,K);
    for(unsigned i=K;i<s.size();i++){
        updateMap(dict,seed,s[i]);
        seed=seed.substr(1)+s[i];
    }

    string reuslt=random_parse(dict);
    cout<<"================="<<endl;
    cout<<reuslt<<endl;
}
