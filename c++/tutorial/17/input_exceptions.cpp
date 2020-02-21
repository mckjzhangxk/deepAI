#include <iostream>
int main() {
    using std::cin;
    using std::cout;
    using std::endl;
    using std::flush;
    int sum = 0;
    int input;

    cin.exceptions(std::ios_base::failbit);
    try {
        while (cin >> input) {
            sum += input;
        }
    } catch (const std::ios_base::failure& e) {
        std::cerr << e.what() << '\n';
    }

    cout << "last valued entered =" << input << endl;
    cout << "sum=" << sum << endl;

    cout << "eof:" << cin.eof() << "bad:" << cin.bad() << "fail:" << cin.fail() << endl;
}