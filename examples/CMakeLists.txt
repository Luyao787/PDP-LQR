cmake_minimum_required(VERSION 3.21)

# # Example executables
# add_executable(simple_example simple_example.cpp)
# target_link_libraries(simple_example pdpLQR)

add_executable(test_example test_example.cpp)
target_link_libraries(test_example pdpLQR)