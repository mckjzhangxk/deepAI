#include <iostream>
#include <cstring>
using namespace std;


/**

	This lession talk about the dymatic allocate and deallocate 
	in c++.


	new() turn a point,delete release the point
	deconstructor: ~slist()


	Let's implement a one-way link to demostration the purpose
*/

///only to store a string
class slistelem{

public:
	//slight complex here,insert this before after,and copy the data into x
	slistelem(char *x,slistelem* after){
		this->data=new char[255];
		strcpy(this->data,x);
		this->next=after;
	}

	//output the content data
	friend ostream & operator<<(ostream &out,slistelem * item){
		cout<<item->data;
		return out;
	}	

	slistelem * nextItem(){return this->next;}
	
	//recollection the memory of this->data
       ~slistelem(){
		cout<<"remove "<<this->data<<endl;
		delete this->data;	
	}
private:
	char * data;
	slistelem * next;
};

class slist{
public:
	slist():head(0){}

	void prepend(char * d){
		//first get a new item
		slistelem *temp=new slistelem(d,this->head);
		this->head=temp;
	}

	//release every slistElem by invoke its deconstructor
	~slist(){
		cout<<"remove slist"<<endl;

		slistelem* p=this->head;
		while(p){
			slistelem *c=p;
			p=p->nextItem();
			// call delete c mean 
			//1)invoke c's deconstructor,let c to release all resource 
			//he allocation.
			//2)recollection the memory allocate to c
			delete c;
		}
	}


	//iteration output every item,notice l is a point type
	//because the caller is a point ,not a slist 'entity' :(
	friend ostream & operator<<(ostream &out,slist* l){
		slistelem* p=l->head;
		while(p){
			out<<p<<endl;
			p=p->nextItem();
		}
		return out;
		
	}

private: slistelem * head;
};

int main(int argc,char* argv[]){
	slist* p_list=new slist();
	p_list->prepend("Hello World");
	p_list->prepend("This is my C++ turtorial");
	p_list->prepend("Thanks");

	cout<<p_list<<endl;
	
	delete p_list;
	return 0;
}
