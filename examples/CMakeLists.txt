cmake_minimum_required(VERSION 3.21)

# # Example executables
add_executable(clqr_example clqr_example.cpp)
target_link_libraries(clqr_example pdpLQR)

add_executable(lqr_example lqr_example.cpp)
target_link_libraries(lqr_example pdpLQR)