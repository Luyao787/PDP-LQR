cmake_minimum_required(VERSION 3.21)

# Find or fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# JSON support (needed for loading reference data)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Create test executable
add_executable(test_lqr
    test_lqr.cpp
)

# Set target properties
target_link_libraries(test_lqr
    PRIVATE
    pdpLQR::pdpLQR
    gtest
    gtest_main
    nlohmann_json::nlohmann_json
)

target_include_directories(test_lqr
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Set compiler flags
target_compile_options(test_lqr PRIVATE -Wall -Wextra -Wconversion -Wpedantic)

include(GoogleTest)
gtest_discover_tests(test_lqr)
