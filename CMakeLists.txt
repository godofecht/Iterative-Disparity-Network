cmake_minimum_required(VERSION 3.13)  # CMake version check
project(DisparityNetwork)             # Create project "Disparity Network"
set(CMAKE_CXX_STANDARD 17)            # Enable c++17 standard


#Clean this up and separate into HEADERS, COMPILATION_UNITS, ETC
set (SOURCES 
NN.h Network.h dispnet.h disparityNet.h filesystem.h model.h utils.h image1.csv image2.csv
)



add_executable(DisparityNetwork ${SOURCES})

target_sources(DisparityNetwork PUBLIC ${SOURCES} main.cpp NN.cpp Network.cpp)


configure_file(image1.csv Debug/image1.csv COPYONLY)
configure_file(image2.csv Debug/image2.csv COPYONLY)
configure_file(image1.csv Release/image1.csv COPYONLY)
configure_file(image2.csv Release/image2.csv COPYONLY)
