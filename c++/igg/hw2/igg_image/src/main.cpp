#include <iostream>
#include <string>
#include "image.h"
using std::cin;
using std::cout;
using std::endl;
using std::string;

std::string get_input(const std::string prompt = "") {
    std::string s;
    cout << prompt << endl;
    cin >> s;
    return s;
}

int main(int argc, char* argv[]) {
    string filename;
    igg::Image img;

    while (true) {
        int chioce;
        cout << "1.read file" << endl;
        cout << "2.write file" << endl;
        cout << "3.upscale image" << endl;
        cout << "4.downscale image" << endl;
        cout << "5.exit" << endl;
        cin >> chioce;
        switch (chioce) {
            case 1:
                filename = get_input("input read file name:");
                if (img.FillFromPgm(filename)) {
                    cout << "read success" << endl;
                }
                break;
            case 2:
                filename = get_input("input output file name:");
                img.WriteToPgm(filename);
                break;
            case 3:
                img.UpScale(2);
                break;
            case 4:
                img.DownScale(2);
                break;
            case 5:
                exit(0);
            default:
                exit(0);
        }
    }
}