#pragma once

#include <vector>
#include <cassert>
#include "lqr/typedefs.hpp"
#include "lqr/status.hpp"

namespace lqr {

struct LQRInfo
{
    LQRStatus status;
    
    int iteration_count;
    
    scalar prim_res_norm;
    scalar dual_res_norm;
};

struct LQRResults
{
    std::vector<VectorXs> xs;
    std::vector<VectorXs> us;
    std::vector<VectorXs> lambdas; // dual variables associated with dynamics constraints
    std::vector<VectorXs> ys;   // dual variables associated with other constraints

    scalar optimal_LQR_cost = LQR_INFTY;

    void reset(int nx, int nu, std::vector<int> ncs, int horizon)
    {
        assert(ncs.size() == horizon + 1);

        xs.resize(horizon + 1);
        us.resize(horizon);
        lambdas.resize(horizon + 1);
        ys.resize(horizon + 1);
        for (int k = 0; k < horizon; ++k) {
            xs[k].resize(nx);
            xs[k].setZero();
            us[k].resize(nu);
            us[k].setZero();
            lambdas[k].resize(nx);
            lambdas[k].setZero();
            ys[k].resize(ncs[k]);
            ys[k].setZero();
        }
        xs[horizon].resize(nx);
        xs[horizon].setZero();
        lambdas[horizon].resize(nx);
        lambdas[horizon].setZero();
        ys[horizon].resize(ncs[horizon]);
        ys[horizon].setZero();

        optimal_LQR_cost = LQR_INFTY;
    }

};


}