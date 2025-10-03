#pragma once

// Core types and utilities
#include "lqr/typedefs.hpp"
#include "lqr/settings.hpp"
#include "lqr/status.hpp"
#include "lqr/results.hpp"

// Model and base classes
#include "lqr/lqr_model.hpp"
#include "lqr/lqr_solver_base.hpp"

// Kernels
#include "lqr/clqr_kernel.hpp"
#include "lqr/clqr_parallel_kernel.hpp"

// Solvers
#include "lqr/clqr_solver.hpp"
#include "lqr/clqr_parallel_solver.hpp"

namespace lqr {
    // Convenient type aliases for common use cases
    using SerialCLQRSolver   = CLQRSolver<CLQRKernelData>;
    using ParallelCLQRSolver = CLQRParallelSolver<ParallelCLQRKernelData>;
}