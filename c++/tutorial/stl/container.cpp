#include<iostream>
#include<string>
#include<vector>
#include<deque>
#include<list>
#include<algorithm>

using namespace std;

int main(int argc, char const *argv[])
{
    
/*
 * Vector
 */
   vector<int> vec;   // vec.size() == 0
   if (vec.empty()) { cout << "Not possible,"; }
   vec.push_back(4);vec.push_back(1);vec.push_back(8);  // vec: {4, 1, 8};    vec.size() == 3

// Vector specific operations:
   cout << vec[2];     // 8  (no range check)
   cout << vec.at(2);  // 8  (throw range_error exception of out of range)
 

   for (vector<int>::iterator itr = vec.begin(); itr!= vec.end(); ++itr)
      cout << *itr << " ";  
   for (auto it: vec)    // C++ 11
      cout << it << " ";

   // Vector is a dynamically allocated contiguous array in memory
   int* p = &vec[0];   p[2] = 6;
   cout<<endl;
   /* Properties of Vector:
   * 1. fast insert/remove at the end: O(1)
   * 2. slow insert/remove at the begining or in the middle: O(n)
   * 3. slow search: O(n)
   */



/*
 * Deque
 */
   deque<float> deq = { 4, 6, 7 };
   deq.push_front(2);  // deq: {2, 4, 6, 7}
   deq.push_back(3);   // deq: {2, 4, 6, 7, 3}

   // Deque has similar interface with vector
   cout << deq[1];  // 4
   cout<<endl;
/* Properties:
 * 1. fast insert/remove at the begining and the end;
 * 2. slow insert/remove in the middle: O(n)
 * 3. slow search: O(n)
 */

   list<int> mylist={1,2,3,4};
   list<int>::iterator iter=find(mylist.begin(),mylist.end(),3);
   // mylist.insert(iter,{5,5,5});
   mylist.erase(iter);
   
   /* Properties:
 * 1. fast insert/remove at any place: O(1)
 * 2. slow search: O(n)
 * 3. no random access, no [] operator.
 * 
 * mylist1.splice(itr, mylist2, itr_a, itr_b );   // O(1)
 */


    return 0;
}
