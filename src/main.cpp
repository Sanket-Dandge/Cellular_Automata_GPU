#include "cyclic_ca.hpp"
#include "forest_fire.hpp"
#include "game_of_life.hpp"
#include "wireworld.hpp"
#include <cxxopts.hpp>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace std;

struct ScopedTimer {
  private:
    string label;
    chrono::high_resolution_clock::time_point start;

  public:
    ScopedTimer(const string &lable) : label(lable), start(chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = chrono::high_resolution_clock::now();
        auto dur = chrono::duration_cast<chrono::microseconds>(end - start).count();
        cout << label << ": " << dur << " μs" << endl;
    }
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("program", "Run different automatons");

    // clang-format off
    options.add_options()
        ("b,benchmark", "Run in benchmark mode")
        ("s,snapshot-interval",
         "Number of frames to take snapshot after, (ignored in benchmark mode)",
         cxxopts::value<int>()->default_value("10"))
        ("a,automaton", "Atomaton to run (gol/cca/ww/ff)", cxxopts::value<string>()->default_value("gol"))
        ("c,config", "Config file to read (only for gol)", cxxopts::value<string>()->default_value(""))
        ("k,kernel", "Kernel to use (base/lut/pc)", cxxopts::value<string>()->default_value("base"))
        ("g,generations", "Number of generations to run simulation for", cxxopts::value<int>()->default_value("1024"))
        ("h,help", "Print usage");
    // clang-format on

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        cout << options.help();
        return 0;
    }

    int snapshot_interval = result["snapshot-interval"].as<int>();
    int generations = result["generations"].as<int>();
    string automaton = result["automaton"].as<string>();
    string config_file = result["config"].as<string>();
    string kernel = result["kernel"].as<string>();
    Implementation impl = BASE;

    // Validate automaton value
    const set<string> valid_automatons = {"gol", "cca", "ww", "ff"};
    if (valid_automatons.find(automaton) == valid_automatons.end()) {
        cerr << "Invalid automaton: " << automaton << ". Valid options are: gol, cca, ww, ff."
             << endl;
        return 1;
    }
    if ((automaton != "gol" || automaton != "ww") && result.count("config") != 0) {
        cerr << "Config option is valid only for GoL & WW" << endl;
    }

    const set<string> valid_kernels = {"base", "lut", "pc"};
    if (valid_kernels.find(kernel) == valid_kernels.end()) {
        cerr << "Invalid kernel: " << kernel << ". Valid options are: base, lut, pc." << endl;
        return 1;
    }
    impl = unordered_map<string, Implementation>(
        {{"base", BASE}, {"lut", LUT}, {"pc", PACKET_CODING}})[kernel];

    if (result.count("snapshot_interval") && result.count("benchmark")) {
        cerr << "WARNING: Running in benchmark mode; ignoring snapshot interval" << endl;
        snapshot_interval = generations;
    }

    // Your logic based on parsed options would go here...
    if (automaton == "gol") {
        GameOfLife life =
            result.count("config") ? GameOfLife(AutomatonConfiguration(config_file)) : GameOfLife();

        {
            ScopedTimer t(format("Iterations-{}", generations));
            life.run(generations, snapshot_interval, impl);
        }
    } else if (automaton == "cca") {
        CyclicCA cca;
        {
            // TODO: Add this option
            ScopedTimer t(format("Iterations-{}", generations));
            cca.run(generations, snapshot_interval);
        }
    } else if (automaton == "ff") {
        {
            ScopedTimer t(format("Iterations-{}", generations));
            switch (impl) {
            case PACKET_CODING: {
                break;
            }
            case BASE: {
                forest_fire_baseline(generations);
                break;
            }
            case LUT: {
                forest_fire_lut(generations);
                break;
            }
            }
        }
    } else if (automaton == "ww") {
        cout << "Running wireworld" << endl;
        WireWorldCA ww = result.count("config") ? WireWorldCA(config_file) : WireWorldCA();
        {
            ScopedTimer t(format("Iterations-{}", generations));
            ww.run(generations, snapshot_interval, impl);
        }
    } else {
        cerr << "Something went wrong, unknown automaton " << automaton << endl;
    }

    return 0;
}
