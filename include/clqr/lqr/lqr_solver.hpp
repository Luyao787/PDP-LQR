#pragma once

#include "clqr/typedefs.hpp"
#include "clqr/lqr_model.hpp"
#include "clqr/lqr/lqr_kernel.hpp"

namespace lqr  {

class LQRSolver {
public:
    LQRSolver(const LQRModel& model);  
    void clear_workspace() { 
        for (auto& data : workspace_) data.set_zero(); 
    }
    void update_problem_data(const std::vector<VectorXs>& ws, 
                             const std::vector<VectorXs>& ys, 
                             const std::vector<VectorXs>& zs,
                             const std::vector<VectorXs>& inv_rho_vecs,
                             const scalar sigma);
    void backward(const std::vector<VectorXs>& rho_vecs);
    void backward_without_factorization(const std::vector<VectorXs>& rho_vecs);
    void forward(const VectorXs& x0, std::vector<VectorXs>& ws);

protected:
    const LQRModel& model_;  
    std::vector<LQRKernelData> workspace_;

};

/* Definitions */
LQRSolver::LQRSolver(const LQRModel& model)
    : model_(model)
{
    workspace_.reserve(model.N + 1);
    for (int k = 0; k < model.N; ++k) {
        workspace_.emplace_back(model.n, model.m, model.nodes[k].n_con);
    }
    workspace_.emplace_back(model.n, 0, model.nodes.back().n_con, true);
}

void LQRSolver::update_problem_data(const std::vector<VectorXs>& ws, 
                                    const std::vector<VectorXs>& ys, 
                                    const std::vector<VectorXs>& zs,
                                    const std::vector<VectorXs>& inv_rho_vecs, const scalar sigma) {
    for (int k = 0; k < model_.N + 1; ++k) {
        auto& kpoint = model_.nodes[k];
        workspace_[k].H = kpoint.H;
        workspace_[k].H.diagonal().array() += sigma;
        workspace_[k].h = kpoint.h;
        workspace_[k].h.noalias() -= sigma * ws[k];
        if (kpoint.n_con > 0) {
            workspace_[k].g = zs[k];
            workspace_[k].g.noalias() -= inv_rho_vecs[k].cwiseProduct(ys[k]);
        }
    }
}

void LQRSolver::backward(const std::vector<VectorXs>& rho_vecs) {
    LQRKernel::terminal_step_with_factorization(model_.nodes.back(), rho_vecs.back(), workspace_.back());
    for (int k = model_.N - 1; k >= 0; --k) {
        LQRKernel::step_with_factorization(model_.nodes[k], rho_vecs[k], workspace_[k + 1], workspace_[k]);
    }
}

void LQRSolver::backward_without_factorization(const std::vector<VectorXs>& rho_vecs) {
    LQRKernel::terminal_step_without_factorization(model_.nodes.back(), rho_vecs.back(), workspace_.back());
    for (int k = model_.N - 1; k >= 0; --k) {
        LQRKernel::step_without_factorization(model_.nodes[k], rho_vecs[k], workspace_[k + 1], workspace_[k]);
    }
}

void LQRSolver::forward(const VectorXs& x0, std::vector<VectorXs>& ws) {
    ws[0].tail(model_.n) = x0;
    for (int k = 0; k < model_.N; ++k) {
        LQRKernel::forward_step(model_.nodes[k], workspace_[k], ws[k], ws[k + 1]);
    }
}


} // namespace lqr