#include "ppm_iostrategy.h"
#include <fstream>
#include <sstream>
#include <string>
using std::endl;

namespace igg {
bool PpmIoStrategy::Write(const std::string& file_name,
                          const ImageData& data) const {
    std::ofstream fout(file_name);
    fout << "P3" << endl;
    fout << data.cols << "  " << data.rows << endl;
    fout << data.max_val << endl;

    int r, g, b;
    for (int i = 0; i < data.rows * data.cols; i++) {
        r = data.data[0][i];
        g = data.data[1][i];
        b = data.data[2][i];
        fout << "   " << r << "   " << g << "   " << b;
        if (i % 4 == 0)
            fout << endl;
    }
    fout.close();
    return true;
}

ImageData PpmIoStrategy::Read(const std::string& file_name) const {
    ImageData data;

    std::ifstream fin(file_name);
    if (!fin.is_open()) {
        return data;
    }

    data.data.push_back(std::vector<int>());
    data.data.push_back(std::vector<int>());
    data.data.push_back(std::vector<int>());

    char line[255];
    int N = 255;
    fin.getline(line, N);
    //filter comments
    while (fin.getline(line, N)) {
        if (line[0] != '#')
            break;
    }
    //rows,cols
    std::stringstream stream(line);
    stream >> data.cols >> data.rows;

    //maxvalue
    fin.getline(line, N);
    stream.clear();
    stream.str("");
    stream << line;
    stream >> data.max_val;

    int r = 0, b = 0, g = 0;

    while (fin.getline(line, N)) {
        stream.clear();
        stream.str("");
        stream << line;
        while (!stream.eof()) {
            stream >> r >> g >> b;
            data.data[0].push_back(r);
            data.data[1].push_back(g);
            data.data[2].push_back(b);
        }
    }
    fin.close();
    return data;
}

}  // namespace igg
