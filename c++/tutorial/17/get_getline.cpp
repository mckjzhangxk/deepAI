#include <iostream>
int main() {
    using std::cin;
    using std::cout;
    using std::endl;

    int ct = 0;
    char ch;

    cin.get(ch);
    while (ch != '\n') {
        cout << ch;
        cin.get(ch);
    }
    cout << endl;

    cout << "using <<:" << endl;
    cin.get(ch);
    while (ch != '\n') {
        cout << ch;
        cin >> ch;
    }
}