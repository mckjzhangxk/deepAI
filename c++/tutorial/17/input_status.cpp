#include <iostream>
int main() {
    using std::cin;
    using std::cout;
    using std::endl;
    using std::flush;
    int sum = 0;
    int input;

    while (cin >> input) {
        sum += input;
    }
    cout << "last valued entered =" << input << endl;
    cout << "sum=" << sum << endl;

    cout << "eof:" << cin.eof() << "bad:" << cin.bad() << "fail:" << cin.fail() << endl;

    cin.clear();
    cin.get();
    while (cin >> input) {
        sum += input;
    }
    cout << "last valued entered =" << input << endl;
    cout << "sum=" << sum << endl;
}