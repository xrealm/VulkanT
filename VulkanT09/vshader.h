//
// Created by Mao on 2020/4/19.
//

#ifndef VULKANT07_VSHADER_H
#define VULKANT07_VSHADER_H

#include <fstream>
#include <iostream>
#include <vector>

static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

#endif //VULKANT07_VSHADER_H
