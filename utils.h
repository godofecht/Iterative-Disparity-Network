#include <iostream>



#define PRINT(name) print(#name, (name))



void print(const char *name, double value) {
    std::cout<<name<<" "<<value<<std::endl;
}