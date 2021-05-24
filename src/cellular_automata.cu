#include "cellular_automata.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

__device__ auto cuda_count_neighbors(char* board
                                    ,int height
                                    ,int width
                                    ,int row
                                    ,int col) -> int
{
    int up = row-1 > -1 ? row-1 : height-1;
    int down = row+1 < height ? row+1 : 0;
    int left = col-1 > -1 ? col-1 : width-1;
    int right = col+1 < width ? col+1 : 0;

    int living_neighbors = 0;
    if(board[(up*width)+left] == 'X')
    ++living_neighbors;
    if(board[(up*width)+col] == 'X')
    ++living_neighbors;
    if(board[(up*width)+right] == 'X')
    ++living_neighbors;
    if(board[(row*width)+left] == 'X')
    ++living_neighbors;
    if(board[(row*width)+right] == 'X')
    ++living_neighbors;
    if(board[(down*width)+left] == 'X')
    ++living_neighbors;
    if(board[(down*width)+col] == 'X')
    ++living_neighbors;
    if(board[(down*width)+right] == 'X')
    ++living_neighbors;

    return living_neighbors;
}

__global__ auto cuda_clear_board(char* board
                                ,int height
                                ,int width) -> void
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx >= (height * width))
        return;
    board[thread_idx] = ' ';
}

__global__ auto cuda_set_cell_status(char* board
                                    ,int height
                                    ,int width) -> void
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= height && col >= width)
        return;
    
    int living_neighbors = cuda_count_neighbors(board
                                               ,height, width
                                               ,row, col);
    if(board[(row*width)+col] == 'X'){
        if(living_neighbors < 2 || living_neighbors > 3){
            board[(row*width)+col] = ' ';
        }
    }else{
        if(living_neighbors == 3){
            board[(row*width)+col] = 'X';
        }
    }
    
    
}

CellularAutomata::CellularAutomata(std::string init_file_path)
{
    get_window_size();
    board = Board(window.ws_row, window.ws_col);
    board.allocate_memory();
    initialize_board(init_file_path);
}

auto CellularAutomata::run() -> void
{
    while(true){
        display_board();
        generate_next_board();
        usleep(250000);
        //while(std::cin.get() != '\n');
    }
}

auto CellularAutomata::get_window_size() -> void
{
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &window);
}

auto CellularAutomata::initialize_board(std::string &init_file_path) -> void
{
    clear_board();
    std::vector<std::vector<int>> positions;
    extract_living_cells(init_file_path, positions);
    set_living_cells(positions);
    board.copy_host_to_device();
}

auto CellularAutomata::clear_board() -> void
{
    dim3 block_size(256);
    dim3 block_count((board.shape.width * board.shape.height + block_size.x - 1) / block_size.x);
    cuda_clear_board<<<block_count, block_size>>>(board.d_board.get()
                                                 ,board.shape.height
                                                 ,board.shape.width);
    cudaDeviceSynchronize();
    board.copy_device_to_host();
}

auto CellularAutomata::extract_living_cells(std::string &init_file_path
                                           ,std::vector<std::vector<int>> &positions) -> void
{
    std::ifstream file;
    file.open(init_file_path.c_str());
    if(!file.is_open()){
        std::cout << "New File Path" << std::endl;
    }

    std::string line;
    while(std::getline(file, line)){
        if(line.empty()) 
            continue;

        std::istringstream stream(line);
        std::string lineStream;
        std::string::size_type size;
        std::vector<int> position;

        while(std::getline(stream, lineStream, ',')){
            position.push_back(std::stoi(lineStream, &size));
        }

        positions.push_back(position);
    }
}

auto CellularAutomata::set_living_cells(std::vector<std::vector<int>> &positions) -> void
{
    for(auto p: positions){
        int row(p[0]), col(p[1]);
        board[(row * board.shape.width) + col] = 'X';
    }
}

auto CellularAutomata::generate_next_board() -> void
{
    dim3 block_size((board.shape.width/2)+1,2);
    dim3 block_count(2,(board.shape.width/2)+1);
    cuda_set_cell_status<<<block_count, block_size>>>(board.d_board.get()
                                                     ,board.shape.height
                                                     ,board.shape.width);
    cudaDeviceSynchronize();
    board.copy_device_to_host();
}

auto CellularAutomata::display_board() -> void
{
    for(auto row=0; row<board.shape.height; row++){
        for(auto col=0; col<board.shape.width; col++){
            std::cout << board[(row*board.shape.width) + col];
        }
        std::cout << '\n';
    }
}
