// src/program/segment_plan_cli.cpp - [claude] planner step 1: the pure module's legacy_plan() on recorded parts, for the
// check "legacy_plan() equals the engine's logged plan on every request" (results/planner-20261112/step1/logplans.py).
// stdin, one part per line:  key=value ... (thresholds and facts; unknown keys are an error)
// stdout, one line per part: path lend_path want head read_chunk
// Build: c++ -std=c++17 -Iinclude src/program/segment_plan_cli.cpp -o segment_plan_cli
#include "strata/program/segment_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace sp = strata::program::segment_plan;

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        sp::Thresholds t;
        sp::PartFacts f;
        std::istringstream in(line);
        std::string kv;
        bool bad = false;
        while (in >> kv) {
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) { bad = true; break; }
            const std::string k = kv.substr(0, eq);
            const long long v = std::atoll(kv.c_str() + eq + 1);
            if (k == "tp_on") t.tp_on = v != 0;
            else if (k == "tp_lo") t.tp_lo = v;
            else if (k == "tp_hi") t.tp_hi = v;
            else if (k == "tp_chunk") t.tp_chunk = v;
            else if (k == "dp_min") t.dp_min = v;
            else if (k == "pipe_chunk") t.pipe_chunk = v;
            else if (k == "pipe_half_min") t.pipe_half_min = v;
            else if (k == "pipe_half_max") t.pipe_half_max = v;
            else if (k == "equal") t.equal = v != 0;
            else if (k == "equal_tail_max") t.equal_tail_max = v;
            else if (k == "a0") f.a0 = v;
            else if (k == "b0") f.b0 = v;
            else if (k == "cap") f.cap = v;
            else if (k == "stream_all_min") f.stream_all_min = v;
            else if (k == "idx_block") f.idx_block = v;
            else if (k == "dp_on") f.dp_on = v != 0;
            else if (k == "layer_split") f.layer_split = v != 0;
            else if (k == "split_small") f.split_small = v;
            else if (k == "split_small_max") f.split_small_max = v;
            else { bad = true; break; }
        }
        if (bad) { std::printf("ERR %s\n", kv.c_str()); continue; }
        const sp::SegmentPlan s = sp::legacy_plan(f, t);
        std::printf("%s %s %lld %lld %lld\n", sp::path_name(s.path), sp::path_name(s.lend_path), (long long) s.want,
                    (long long) s.head, (long long) s.read_chunk);
    }
    return 0;
}
