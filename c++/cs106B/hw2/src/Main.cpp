#include <iostream>
#include "console.h"
#include "simpio.h"
#include "random.h"

using namespace std;
void markov_main();
void play_maze();

int main() {
  cout << "chioce [1-2]" << endl;
  cout<<"1.Markov models of language"<<endl;
  cout<<"2.maze"<<endl;
  int chioce=getInteger();
  switch (chioce) {
    case 1:
          markov_main();
          break;
    case 2:
        play_maze();
        break;
  }


  return 0;

}
