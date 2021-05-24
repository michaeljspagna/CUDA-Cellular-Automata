#include "src/cellular_automata.hpp"

#include <iostream>
#include <string>

int main()
{
    char choice;
    std::string path;
    
    std::cout 
        << "1-3 for sample, 4 for your own file, or 5 to quit:"; 
    std::cin >> choice;

    switch (choice)
    {
    case '1':
        path = "samples/sample1.txt";
        break;
    case '2':
        path = "samples/sample2.txt";
        break;
    case '3':
        path = "samples/sample3.txt";
        break;
    case '4':
        std::cout << "Enter Path: ";
        std::cin >> path;
        break;
    case '5':
        return 1;
    default:
        std::cout << "invalid" << std::endl;
        return -1;
    }

    CellularAutomata ca(path);
    ca.run();
}