/*
 * CS 106B/X Sample Project
 * last updated: 2018/09/19 by Marty Stepp
 *
 * This project helps test that your Qt Creator system is installed correctly.
 * Compile and run this program to see a console and a graphical window.
 * If you see these windows, your Qt Creator is installed correctly.
 */

#include <iostream>
#include "console.h"
#include "ginteractors.h" // for GWindow
#include "gwindow.h" // for GWindow
#include "simpio.h"  // for getLine
#include "vector.h"  // for Vector
#include "program.h"

using namespace std;

bool canMakeSum(Vector<int> &num,int target);
int main() {
    cout << "chioce [1-2]" << endl;
    cout<<"1.warnup"<<endl;
    cout<<"2.12Steps"<<endl;
    cout<<"3. Ruler of the world"<<endl;
    cout<<"4. Every vote counts"<<endl;
    cout<<"5. Puzzle"<<endl;

    int chioce=getInteger();
    Vector<int> v={3,7,1,8,-3};
    switch (chioce) {
      case 1:
            cout<<canMakeSum(v,54);
            break;
      case 2:
          cout<<CountWays(4)<<endl;
          break;
      case 3:
            DrawRuler(40,600,1000,600);
            break;
      case 4:
//        cout<<CountCriticalVotes({4,2,7,4},0)<<endl;
//        cout<<CountCriticalVotes({4,2,7,4},1)<<endl;
//        cout<<CountCriticalVotes({4,2,7,4},2)<<endl;
//        cout<<CountCriticalVotes({4,2,7,4},3)<<endl;
        cout<<CountCriticalVotes({9,9,7,3,1,1},0)<<endl;
        cout<<CountCriticalVotes({9,9,7,3,1,1},1)<<endl;
        cout<<CountCriticalVotes({9,9,7,3,1,1},2)<<endl;
        cout<<CountCriticalVotes({9,9,7,3,1,1},3)<<endl;
        cout<<CountCriticalVotes({9,9,7,3,1,1},4)<<endl;
        cout<<CountCriticalVotes({9,9,7,3,1,1},5)<<endl;
        break;
       case 5:
//        cout<<solable(0,{3,6,4,1,3,4,2,5,3,0})<<endl;
        cout<<solable(0,{3,1,2,3,0});
        break;
    }


    return 0;

    return 0;
}
