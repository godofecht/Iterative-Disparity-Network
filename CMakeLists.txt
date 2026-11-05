cmake_minimum_required(VERSION 3.16)
project(DisparityNetwork LANGUAGES CXX)

add_library(disparity_core src/DisparityNetwork.cpp)
target_include_directories(disparity_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(disparity_core PUBLIC cxx_std_17)

add_executable(DisparityNetwork src/main.cpp)
target_link_libraries(DisparityNetwork PRIVATE disparity_core)

foreach(data_file IN ITEMS image1.csv image2.csv disparity.csv)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${data_file}" "${CMAKE_CURRENT_BINARY_DIR}/${data_file}" COPYONLY)
endforeach()

include(CTest)
if(BUILD_TESTING)
    add_executable(disparity_tests tests/network_tests.cpp)
    target_link_libraries(disparity_tests PRIVATE disparity_core)
    add_test(NAME gradients_and_training COMMAND disparity_tests)
    add_test(NAME training_smoke COMMAND DisparityNetwork --epochs 3 --output-dir smoke-results)
    set_tests_properties(training_smoke PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()
