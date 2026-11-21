// tools/test/planner_rows_dump.cpp - [claude] B2: print the rows the engine's own parser (planner_rows.hpp) reads from a --planner
// table, one "KEY VALUE" line each, in file order (the CPU parity proof diffs a generated table against the live one this way).
// Build: c++ -O2 -std=c++17 -Iinclude tools/test/planner_rows_dump.cpp -o planner_rows_dump
#include "strata/program/planner_rows.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: planner_rows_dump TABLE\n"; return 2; }
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf();
    std::vector<strata::program::planner_rows::Row> rows; std::string err;
    if (!f || !strata::program::planner_rows::parse(ss.str(), rows, err)) { std::cerr << "refused: " << err << "\n"; return 1; }
    for (const auto& r : rows) std::cout << r.key << " " << r.value << "\n";
    return 0;
}
