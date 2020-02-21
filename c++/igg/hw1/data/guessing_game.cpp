#include <iostream>
#include <cstdlib>
#include<ctime>
int main(int argc,char* argv[]){
    using namespace std;

    srand(time(nullptr));
    
    int answer=rand()%100;
    int guess=0;
    while (true)
    {
        cout<<"Input your guess:";
        cin>>guess;
        if(guess==answer){
            cout<<"answer is "<<guess<<endl;
            break;
        }else if(guess<answer){
            cout<<"answer is greater than "<<guess<<endl;
        }else
        {
            cout<<"answer is less than "<<guess<<endl;
        }
        
    }
    return 0;
}