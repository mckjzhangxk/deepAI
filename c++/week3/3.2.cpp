#include <iostream>
using namespace std;
/*
	This class mainly talk about the use of 
	initializer.

        initializer sometime is necessary when 
        you want to initial a const,notice assign 
        value to const is forbid by c++ compiler

*/
class point{
public:
        //fancy synax of c++,using initializer
        point(double x=0,double y=0):x(x),y(y),PI(55){
		//feel free to uncomment to see the result		
		//PI=55;
		//this->y=11;
	}
        // overload <<
        friend ostream & operator<<(ostream &out,point &p){
                out<<"("<<p.x<<","<<p.y<<")";
		out<<p.PI;
                return out;
        }
private:
        double x;
        double y;
	const int PI;
};

int main(int argc,char * argv[]){
        point a(3,4);
        cout<<a<<endl;
        return 0;
}
