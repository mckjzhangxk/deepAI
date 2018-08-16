#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
using namespace std;

const int DICE_SIDE=6;
inline int rise_dice(){
	return rand()%DICE_SIDE +1;
}

int main(int argc,char* argv[]){
	int* outcome=new int[13];
	int trial=0;
	std::cout <<"\n Enter # of trials:";
	std::cin >>trial;
	
	srand(clock());
	for(int i=0;i<trial;i++){		
		outcome[rise_dice()+rise_dice()]++;
	}
	for(int i=2;i<=12;i++){
		//printf("prob sum=%d is%.3f\n",i,((double)outcome[i])/trial);
		cout<<"the prob of sum="<<i<<" is"<< static_cast<double>(outcome[i])/trial<<endl;
	}
	return 0;
}
