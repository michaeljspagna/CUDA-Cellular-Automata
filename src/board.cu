#include "board.hpp"

Board::Board(int height, int width)
    : shape(height, width)
    , d_board(nullptr), h_board(nullptr)
    , d_allocated(false), h_allocated(false)
{}

auto Board::allocate_memory() -> void
{
    device_allocate_memory();
    host_allocate_memory();
}

auto Board::allocate_memory_if_not_allocated(Shape shape) -> void
{
    if(!d_allocated && !h_allocated){
        this->shape = shape;
        allocate_memory();
    }
}

auto Board::device_allocate_memory() -> void
{
    if(!d_allocated){
        char* d_memory = nullptr;
        cudaMalloc(&d_memory,shape.width * shape.height * sizeof(char));
        d_board = std::shared_ptr<char> (d_memory,[&](char* ptr){ cudaFree(ptr); });
        d_allocated = true;
    }
}

auto Board::host_allocate_memory() -> void
{
    if(!h_allocated){
        h_board = std::shared_ptr<char>(new char[shape.width * shape.height],[&](char* ptr){ delete[] ptr; });
        h_allocated = true;
    }
}

auto Board::copy_device_to_host() -> void
{
    if(d_allocated && h_allocated){
        cudaMemcpy(h_board.get(), d_board.get()
                  ,shape.width * shape.height * sizeof(char)
                  ,cudaMemcpyDeviceToHost);
    }
}

auto Board::copy_host_to_device() -> void
{
    if(d_allocated && h_allocated){
        cudaMemcpy(d_board.get(), h_board.get()
                  ,shape.width * shape.height * sizeof(char)
                  ,cudaMemcpyHostToDevice);
    }
}

auto Board::operator[](const int index) -> char&
{
    return h_board.get()[index];
}

auto Board::operator[](const int index) const -> const char&
{
    return h_board.get()[index];
}
