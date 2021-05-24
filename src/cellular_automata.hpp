#ifndef CELLULAR_AUTOMATA_HPP
#define CELLULAR_AUTOMATA_HPP

#include <string>
#include <sys/ioctl.h>
#include <vector>
#include "board.hpp"

struct CellularAutomata
{
private:
    Board board;
    winsize window;

    auto get_window_size() -> void;
    auto initialize_board(std::string &init_file_path) -> void;
    auto clear_board() -> void;
    auto extract_living_cells(std::string &init_file_path
                             ,std::vector<std::vector<int>> &positions) -> void;
    auto set_living_cells(std::vector<std::vector<int>> &positions) -> void;

    auto generate_next_board() -> void;
    auto display_board() -> void;

public:
    CellularAutomata(std::string init_file_path);

    auto run() -> void;
};

#endif /* CELLULAR_AUTOMATA_HPP */