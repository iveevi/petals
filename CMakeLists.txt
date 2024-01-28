cmake_minimum_required(VERSION 3.20)

project(petals CXX)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

set(PETAL_SOURCES source/tensor.cpp)

add_library(petal SHARED ${PETAL_SOURCES})

add_executable(features-tests tests/features/main.cpp)
add_executable(gradients-tests tests/gradients/main.cpp)

target_link_libraries(features-tests fmt petal)
target_link_libraries(gradients-tests fmt petal)
