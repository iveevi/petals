project(petals CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(main main.cpp)

target_link_libraries(main fmt)
