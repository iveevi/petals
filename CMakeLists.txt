cmake_minimum_required(VERSION 3.20)

project(petals CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG master)
FetchContent_MakeAvailable(fmt)

set(PETAL_SOURCES
	source/tensor.cpp
	source/gradients.cpp
	source/kernels.cpp
	source/resource.cpp
	source/composition.cpp)

add_library(petal SHARED ${PETAL_SOURCES})

include_directories(include)
add_executable(features-tests tests/features/main.cpp)
add_executable(gradients-tests tests/gradients/main.cpp)
add_executable(mnist tests/mnist/main.cpp)

target_link_libraries(features-tests petal fmt::fmt)
target_link_libraries(gradients-tests petal fmt::fmt)
target_link_libraries(mnist petal fmt::fmt)
