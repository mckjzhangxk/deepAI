#include <iostream>
using namespace std;
typedef enum  days{SUN,MON,TUE,WED,THE,FRI,SAT} days;
/*
	1)overload ++ operator :days operator++(days)
	2)overload << operator :ostream &  operator<<(ostream &,const days&)
*/

/*

	return is a reference because we want it associate with screen
	const mean non-mutable of input
*/
ostream & operator<<(ostream & out,const days & d){
	switch(d){
		case SUN:out<<"SUN";break;
		case MON:out<<"MON";break;
		case TUE:out<<"TUE";break;
		case WED:out<<"WED";break;
		case THE:out<<"THE";break;
		case FRI:out<<"FRI";break;
		case SAT:out<<"SAT";break;
	}
	return out;
}
/*
	version 1:overload ++
	d++ will return next day,but d won't change
*/
inline days operator++(days d){
	return static_cast<days>((static_cast<int>(d)+1)%7);
}


/*
	version 2:overload ++
	d++ will return next day,but d will change
*/
/*inline days& operator++(days& d){
	d=static_cast<days>((static_cast<int>(d)+1)%7); //change value of d
	return d;
}*/
int main(int argc,char **argv){
	days d=FRI,e;
	cout<<"day d++ is:"<<(++d)<<endl;
	cout<<"day d is:"<<d<<endl;
	e=d;
	++e;
	cout<<"day e is:"<<e<<endl;
	cout<<"day d is:"<<d<<endl;
	return 0;
}

