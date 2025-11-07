# PDP-LQR

This repository contains the code for the paper titled "Parallel Dynamic Programming for Conic
Linear Quadratic Control".

## Overview

PDP-LQR is a library for solving Linear Quadratic (LQ) problems using parallel dynamic programming algorithms. The full code will be released soon.

<!-- ## Features -->

## Dependencies

- **CMake** (3.21 or later)
- **C++17 compiler** 
- **Eigen3** (3.3 or later)
```bash
sudo apt update
sudo apt install libeigen3-dev
```
- **OpenMP** (for parallel algorithms)
- **QDLDL** 
```bash
git clone https://github.com/osqp/qdldl.git
cd qdldl
mkdir build && cd build
cmake ..
cmake --build .
cmake --install .
```

## Installation

### Using CMake

```bash
git clone https://github.com/Luyao787/PDP-LQR.git
cd PDP-LQR
mkdir build && cd build
cmake ..
cmake --build .
cmake --install .
```

### Using the library in your project

#### With CMake

Add to your `CMakeLists.txt`:

```cmake
find_package(pdpLQR REQUIRED)
target_link_libraries(your_target pdpLQR::pdpLQR)
```

## Examples

Build and run examples:

```bash
cmake -DBUILD_EXAMPLES=ON ..
cmake --build .
./examples/lqr_example
```

<!-- ## Testing -->

<!-- Build and run tests: -->

<!-- ```bash
cmake -DBUILD_TESTS=ON ..
make
make test
``` -->

