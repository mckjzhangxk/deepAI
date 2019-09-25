#include <iostream>
#include "console.h"
#include "simpio.h"

using namespace std;
void markov_main();

int main() {
  cout << "chioce [1-2]" << endl;
  cout<<"1.Markov models of language"<<endl;
  int chioce=getInteger();
  switch (chioce) {
    case 1:
          markov_main();
          break;
  }
  return 0;

}
