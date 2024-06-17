#include "misc.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>

uint64_t GetTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

long long int _accumulator;

void dbg_count(int val) { _accumulator += val; }

void dbg_print() { std::cout << _accumulator << std::endl; }

// splits a string into a vector of tokens and returns it
std::vector<std::string> split_command(const std::string& command) {
    std::stringstream stream(command);
    std::string intermediate;
    std::vector<std::string> tokens;

    while (std::getline(stream, intermediate, ' ')) {
        tokens.push_back(intermediate);
    }

    return tokens;
}

// returns true if in a vector of string there's one that matches the key
bool Contains(const std::vector<std::string>& tokens, const std::string& key) {
    return std::find(tokens.begin(), tokens.end(), key) != tokens.end();
}
