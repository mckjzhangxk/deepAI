#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void Read(const std::string& file_name) {
    // ImageData data;

    std::ifstream fin(file_name.c_str());
    if (!fin.is_open()) {
        return;
    }
    char line[255];
    int N = 255;
    fin.getline(line, N);
    //filter comments
    while (fin.getline(line, N)) {
        if (line[0] != '#')
            break;
    }
    //rows,cols
    int rows, cols;
    std::stringstream stream;
    stream << line;
    stream >> rows >> cols;

    //maxvalue
    int maxval;
    fin.getline(line, N);
    stream.clear();
    stream.str("");
    stream << line;
    stream >> maxval;

    int id = 0;
    int total = rows * cols;
    int r = 0, b = 0, g = 0;

    while (fin.getline(line, N)) {
        stream.clear();
        stream.str("");
        stream << line;
        while (!stream.eof()) {
            stream >> r >> g >> b;
            // data.data[id].push_back(r);
            // data.data[id].push_back(g);
            // data.data[id].push_back(b);
            ++id;
        }
    }

    std::cout << "xxxxxxxxxxxxxx:" << id << std::endl;
}

int main() {
    using std::cin;
    using std::cout;
    using std::endl;
    using std::string;

    // int c = 0;
    // int cnt = 0;
    // std::stringstream ss;
    // char line[255];
    // while (cin.getline(line, 255)) {
    //     ss << line;

    //     while (ss >> c) {
    //         cnt += 1;
    //     }
    //     cout << ss.eof() << endl;
    //     ss.clear();
    // }

    // cout << "total:" << cnt << endl;
    Read("/home/zxk/PycharmProjects/deepAI1/c++/tutorial/17/build/a.txt");
}