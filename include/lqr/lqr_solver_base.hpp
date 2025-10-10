#pragma once

#include "lqr/typedefs.hpp"

namespace lqr {

class LQRSolverBase {
public:
    LQRSolverBase() = default;
    
    virtual ~LQRSolverBase() = default;

    virtual void backward(scalar sigma) = 0;

    virtual void forward(const VectorXs& x0) = 0;
    
};

} // namespace lqr