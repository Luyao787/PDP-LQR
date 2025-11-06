
cmake_minimum_required(VERSION 3.21)
project(pdpLQR 
    VERSION 1.0.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        # set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()
endif()
message(STATUS "Building ${CMAKE_BUILD_TYPE}")

# Set compiler flags
list(APPEND compiler_flags -Wall -Wextra -Wconversion -Wpedantic)
# list(APPEND compiler_flags -Wall -Wextra -Wconversion -Wpedantic -march=native -O2)
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Find dependencies
# include(FetchContent)
# FetchContent_Declare (
#     tracy
#     GIT_REPOSITORY https://github.com/wolfpld/tracy.git
#     GIT_TAG master
#     GIT_SHALLOW TRUE
#     GIT_PROGRESS TRUE
# )
# FetchContent_MakeAvailable (tracy)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# Find QDLDL
find_path(QDLDL_INCLUDE_DIR qdldl/qdldl.h)
find_library(QDLDL_LIBRARY NAMES qdldl)

if(QDLDL_INCLUDE_DIR AND QDLDL_LIBRARY)
    message(STATUS "Found QDLDL: ${QDLDL_LIBRARY}")
    add_library(QDLDL::QDLDL UNKNOWN IMPORTED)
    set_target_properties(QDLDL::QDLDL PROPERTIES
        IMPORTED_LOCATION "${QDLDL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${QDLDL_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "QDLDL not found. Please install QDLDL or set QDLDL_INCLUDE_DIR and QDLDL_LIBRARY")
endif()

# Create header-only library
add_library(pdpLQR INTERFACE)
add_library(pdpLQR::pdpLQR ALIAS pdpLQR)

# Set target properties
target_include_directories(pdpLQR 
    INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(pdpLQR 
    INTERFACE 
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    QDLDL::QDLDL
    # TracyClient
)

target_compile_features(pdpLQR INTERFACE cxx_std_17)

target_compile_options(pdpLQR INTERFACE ${compiler_flags})

option(ENABLE_LQR_VERBOSE "Enable verbose output for LQR solver" ON)
if(ENABLE_LQR_VERBOSE)
    target_compile_definitions(pdpLQR INTERFACE LQR_VERBOSE)
endif()

# Install targets
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install library
install(
    TARGETS pdpLQR
    EXPORT pdpLQRTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for find_package() support
install(
    EXPORT pdpLQRTargets
    FILE pdpLQRTargets.cmake
    NAMESPACE pdpLQR::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pdpLQR
)

# Create package config files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pdpLQRConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pdpLQRConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pdpLQR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/pdpLQRConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/pdpLQRConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/pdpLQRConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pdpLQR
)

# Export from build tree
export(
    EXPORT pdpLQRTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/pdpLQRTargets.cmake"
    NAMESPACE pdpLQR::
)

# Options for building examples and tests
option(BUILD_EXAMPLES "Build examples" ON)
# option(BUILD_TESTS "Build tests" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# if(BUILD_TESTS)
#     add_subdirectory(tests)
# endif()