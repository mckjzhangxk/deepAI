#include <iostream>
using namespace std;


/**

	Three major topic in this class
	1)how to construct a class, constructor
	2)overload operator such as +,<<
	3)accessor and mutator, class scope
	
	excise:
	4)mix with generic template

*/

class Point{

private: 	
	double x;
	double y;
public: 
	//part 1)
	Point(){
		this->x=0.0;
		this->y=0.0;
	}
	Point(double x,double y){
		this->x=x;
		this->y=y;	
	}

	//part 3)
	double getX(){return this->x;}
	double getY(){return this->y;}
	void setX(double x){this->x=x;}
	void setY(double y){this->y=y;}

	//overload <<
	//we can using point p instead of const point& p, call by value,but it's not efficient
	//no need for construct a new class,and copy values (: 

	//define operation in a class mean self reference is first parameter,out is second,in a other word,
	//you only can declase like :  ostream & operator<<(ostream & out){
	//therefore you call p<<cout instead of  cout<<p ,
	//and key word friend can sove this problem,why?
	friend ostream & operator<<(ostream & out,Point &p){
			out<<"("<< p.x <<","<< p.y <<")";
			return out;
	}

	
	//overload +,notice p1 is second parameter
	Point operator+(const Point& p1){
		Point sum;
		sum.x=p1.x+this->x;
		sum.y=p1.y+this->y;
		return sum;
	}
};
/*
	Define overload operator << globally
*/

	/*ostream & operator<<(ostream & out,Point& p){
			out<<"("<< p.getX() <<","<< p.getY() <<")";
			return out;
	}*/

	

int main(int argc,char **argv){
	Point a;
	Point b(5,6);
	cout<<a<<endl;
	cout<<b<<endl;


	Point c=a+b;
	cout<<c<<endl;
	return 0;
}

