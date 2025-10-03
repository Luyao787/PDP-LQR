#pragma once

namespace lqr {

enum struct LQRStatus {
    LQR_SOLVED = 0,
    LQR_MAX_ITER_REACHED,
    LQR_PRIM_INFEASIBLE,
    LQR_DUAL_INFEASIBLE,
};

} // namespace lqr