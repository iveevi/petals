cmake_minimum_required(VERSION 3.20)

project(petals CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

find_package(OpenMP REQUIRED)

FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG master)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.14.x)
FetchContent_Declare(googlebenchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG main)
FetchContent_MakeAvailable(googletest googlebenchmark)

set(PETAL_SOURCES
	source/tensor.cpp
	source/gradients.cpp
	source/kernels.cpp
	source/resource.cpp
	source/composition.cpp)

add_library(petal SHARED ${PETAL_SOURCES})

include_directories(include
	${fmt_SOURCE_DIR}/include
	${googlebenchmark_SOURCE_DIR}/include
	${googletest_SOURCE_DIR}/googletest/include)

add_executable(features-tests tests/features/main.cpp)
add_executable(gradients-tests tests/gradients/main.cpp)
add_executable(mnist tests/mnist/main.cpp)
add_executable(ops-benchmark tests/benchmark/main.cpp)
# Allocator benchmark

target_link_libraries(features-tests petal fmt::fmt OpenMP::OpenMP_CXX)
target_link_libraries(gradients-tests petal fmt::fmt OpenMP::OpenMP_CXX gtest_main)
target_link_libraries(mnist petal fmt::fmt OpenMP::OpenMP_CXX)
target_link_libraries(ops-benchmark petal fmt::fmt benchmark::benchmark OpenMP::OpenMP_CXX)
