#ifndef BOARD_HPP
#define BOARD_HPP

#include <iostream>
#include <memory>
#include "shape.hpp"

struct Board
{        
private:
    bool d_allocated;
    bool h_allocated;

    auto device_allocate_memory() -> void;
    auto host_allocate_memory() -> void;
    
public:
    Shape shape;
    std::shared_ptr<char> d_board;
    std::shared_ptr<char> h_board;

    Board(int height=-1, int width=-1);

    auto allocate_memory() -> void;
    auto allocate_memory_if_not_allocated(Shape shape) -> void;

    auto copy_device_to_host() -> void;
    auto copy_host_to_device() -> void;

    auto operator[](const int index) -> char&;
    auto operator[](const int index) const -> const char&;

};

#endif /* BOARD_HPP */