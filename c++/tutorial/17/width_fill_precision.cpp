#include <iostream>
int main() {
    using std::cin;
    using std::cout;
    using std::endl;
    using std::flush;

    int N = 11;
    cout.fill('*');
    for (int i = 1; i <= N; i++) {
        cout.width(2);
        cout << i << ":";
        cout.width(4);
        cout << i * i << endl;
    }

    cout << "precision demo" << endl;
    float a = 200.1234;
    float b = 2.1234;

    cout << "a=" << a << ","
         << "b=" << b << endl;
    cout.precision(2);
    cout << "a=" << a << ","
         << "b=" << b << endl;
    cout.precision(3);
    cout << "a=" << a << ","
         << "b=" << b << endl;
}