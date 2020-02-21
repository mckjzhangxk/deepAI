#include <cmath>
#include <iomanip>
#include <iostream>
int main() {
    using namespace std;

    int w1 = 5, w2 = 10, w3 = 10;
    cout << setw(w1) << "N"
         << setw(w2) << "square root"
         << setw(w3) << "fourth root" << endl;

    for (int n = 10; n <= 100; n += 10) {
        float root = sqrt(n);
        float fourth_root = sqrt(n);
        cout << setw(w1) << setfill('.') << n << setfill(' ') << setw(w2) << setprecision(3) << root << setw(w3) << setprecision(4) << fourth_root << endl;
    }
}