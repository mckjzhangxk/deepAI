#include "soundex.h"
#include "simpio.h"
Soundex::Soundex(string & s)
{
    string g=toUpperCase(s);

    char ch=g[0];
    mcodes.add(ch);

    for(int i=1;i<g.size();i++){
        if(i<g.size()-1 &&CODE_DICT[g[i]]==CODE_DICT[g[i+1]])
            continue;
        ch=CODE_DICT[g[i]];
        if(ch=='0') continue;
        mcodes.add(ch);
        if(mcodes.size()==4)
            break;
    }

    int cnt=4-mcodes.size();
    for(int i=0;i<cnt;i++)
        mcodes.add('0');

}
void Soundex::init(){

}
ostream& operator<<(ostream& out,const Soundex & v){
    for(int i=0;i<v.mcodes.size();i++){
        out<<v.mcodes[i];
    }
    return out;
}

void part3(){
    while (true) {
        string name=getLine("Enter surname:");
        Soundex sname(name);
        cout<<"Soundex code for "<<name<<" is "<<sname<<endl;
    }
}
