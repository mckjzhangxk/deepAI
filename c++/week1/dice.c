#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DICE_SIDE 6
#define RISE_DICE (rand()%DICE_SIDE +1)

int main(int argc,char* argv[]){
	int outcome[13]={0,0,0,0,0,0,0,0,0,0,0,0,0};
	int trial=100;

	srand(clock());
	for(int i=0;i<trial;i++){		
		outcome[RISE_DICE+RISE_DICE]++;
	}
	for(int i=2;i<=12;i++){
		printf("prob sum=%d is%.3f\n",i,((double)outcome[i])/trial);
	}
	return 0;
}
