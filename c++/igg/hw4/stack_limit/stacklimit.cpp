#include <iostream>
using namespace std;
//stack have limit 8100K
int main(int argc, char* argv[]) {
    int s = 0;
    while (true) {
        int n = s * 100 * 1024 / sizeof(double);
        double arr[n];
        for (int i = 0; i < n; i++) {
            arr[i] += 1;
        }
        cout << "memory allocation:" << sizeof(double) * n / 1024 << "K" << endl;
        s++;
    }
}