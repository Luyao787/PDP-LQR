cmake_minimum_required(VERSION 3.21)

add_executable(lqr_example lqr_example.cpp)
target_link_libraries(lqr_example pdpLQR)