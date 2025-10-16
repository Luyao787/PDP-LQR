#pragma once

#include <iostream>
#include <iomanip>
#include "clqr/typedefs.hpp"
#include "clqr/lqr/lqr_solver.hpp"
#include "clqr/lqr/lqr_solver_parallel.hpp"
#include "clqr/osclqr_kernel.hpp"
#include "clqr/results.hpp"

namespace lqr {

struct OSCLQRWorkspace {
    std::vector<VectorXs> rho_vecs;
    std::vector<VectorXs> inv_rho_vecs;
    std::vector<OSCLQRKernelData> kernel_data_vec;
    
    std::vector<VectorXs> ws;
    std::vector<VectorXs> ys;
    std::vector<VectorXs> zs;

    OSCLQRWorkspace() = default;

    OSCLQRWorkspace(int n, int m, std::vector<int>& ncs, int N) {
        rho_vecs.resize(N + 1);
        inv_rho_vecs.resize(N + 1);
        kernel_data_vec.reserve(N + 1);
        ws.resize(N + 1);
        ys.resize(N + 1);
        zs.resize(N + 1);
        for (int k = 0; k < N; ++k) {
            int nc = ncs[k];
            kernel_data_vec.emplace_back(n, m, nc);
            rho_vecs[k].resize(nc);
            inv_rho_vecs[k].resize(nc);
            ws[k].resize(n + m);
            ws[k].setZero();
            ys[k].resize(nc);
            ys[k].setZero();
            zs[k].resize(nc);
            zs[k].setZero();
        }
        int nc = ncs.back();
        kernel_data_vec.emplace_back(n, 0, nc, true);
        rho_vecs.back().resize(nc);
        inv_rho_vecs.back().resize(nc);
        ws.back().resize(n);
        ws.back().setZero();
        ys.back().resize(nc);
        ys.back().setZero();
        zs.back().resize(nc);
        zs.back().setZero();
    }

};

class OSCLQRSolver {
public:

    OSCLQRSolver(const LQRModel& model);

    virtual void update_dual_variables(scalar alpha);
    
    void init_rho(const LQRSettings& settings);
    void update_rho(const LQRSettings& settings);

    void compute_residuals(const LQRSettings& settings);
    int check_termination();

    void update_results(LQRResults& results);
    scalar compute_LQR_cost();

    void solve(const VectorXs& x0, const LQRSettings& settings, LQRResults& results);

private:
    const LQRModel& model_;
    OSCLQRWorkspace workspace_;
    std::vector<int> ncs_;
    
    // SerialLQRSolver lqr_solver_;
    LQRParallelSolver lqr_solver_;

    scalar rho_sparse_; 
    scalar rho_estimate_sparse_;
    scalar prim_res_norm_; 
    scalar dual_res_norm_;
    scalar prim_res_norm_rel_; 
    scalar dual_res_norm_rel_;
    scalar eps_prim_; 
    scalar eps_dual_;

    bool is_rho_vec_init_ = false;

};

OSCLQRSolver::OSCLQRSolver(const LQRModel& model)
    : model_(model),
      lqr_solver_(model, 2, true),
      rho_sparse_(0.1), 
      rho_estimate_sparse_(0.1),
      prim_res_norm_(0.0), 
      dual_res_norm_(0.0),
      prim_res_norm_rel_(0.0), 
      dual_res_norm_rel_(0.0),
      eps_prim_(0.0), 
      eps_dual_(0.0) 
{
    for (const auto& kpoint : model.nodes) {
        ncs_.push_back(kpoint.n_con);
    }
    workspace_ = OSCLQRWorkspace(model.n, model.m, ncs_, model.N);
}

void OSCLQRSolver::init_rho(const LQRSettings& settings) {
    rho_sparse_ = settings.rho;
    rho_estimate_sparse_ = settings.rho;
    for (int k = 0; k < model_.N + 1; ++k) {
        workspace_.rho_vecs[k].setConstant(rho_sparse_);
        workspace_.inv_rho_vecs[k].setConstant(1.0 / rho_sparse_);
    }
}

void OSCLQRSolver::update_dual_variables(scalar alpha) {
    for (int k = 0; k < model_.N + 1; ++k) {
        auto& kernel_data = workspace_.kernel_data_vec[k];
        auto& kpoint = model_.nodes[k];
        if (kpoint.n_con > 0) {
            kernel_data.Dw_tmp.noalias() = kpoint.D_con * workspace_.ws[k];
            kernel_data.z_prev = workspace_.zs[k];
            kernel_data.z_tilde.noalias()  = alpha * kernel_data.Dw_tmp;
            kernel_data.z_tilde.noalias() += (1. - alpha) * workspace_.zs[k];

            kernel_data.y_tmp = workspace_.inv_rho_vecs[k].cwiseProduct(workspace_.ys[k]);
            workspace_.zs[k]  = kernel_data.z_tilde + kernel_data.y_tmp;
            workspace_.zs[k]  = workspace_.zs[k].cwiseMax(kpoint.e_lb).cwiseMin(kpoint.e_ub);
            workspace_.ys[k] += workspace_.rho_vecs[k].cwiseProduct(kernel_data.z_tilde - workspace_.zs[k]);
        }
    }
}

void OSCLQRSolver::compute_residuals(const LQRSettings& settings) {
    prim_res_norm_ = 0.0;
    dual_res_norm_ = 0.0;
    prim_res_norm_rel_ = 0.0;
    dual_res_norm_rel_ = 0.0;

    for (int k = 0; k < model_.N + 1; ++k) {
        auto& kernel_data = workspace_.kernel_data_vec[k];
        auto& kpoint = model_.nodes[k];
        OSCLQRKernel::residual_step(kpoint, 
                                    workspace_.ys[k], workspace_.zs[k], workspace_.rho_vecs[k],
                                    kernel_data,
                                    prim_res_norm_, dual_res_norm_,
                                    prim_res_norm_rel_, dual_res_norm_rel_);
    }

    eps_prim_ = settings.eps_abs + prim_res_norm_rel_ * settings.eps_rel;
    eps_dual_ = settings.eps_abs + dual_res_norm_rel_ * settings.eps_rel;
}

void OSCLQRSolver::update_results(LQRResults& results) {
    int nx = model_.n;
    int nu = model_.m;
    int N  = model_.N;
    for (int k = 0; k < N; ++k) {
        results.xs[k] = workspace_.ws[k].tail(nx);
        results.us[k] = workspace_.ws[k].head(nu);
        // results.lambdas[k] = 
        if (model_.nodes[k].n_con > 0) { results.ys[k] = workspace_.ys[k]; }
    }
    results.xs[N] = workspace_.ws[N].tail(nx);
    // results.lambdas[N] = workspace_.kernel_data_vec[N].lambda;
    if (model_.nodes[N].n_con > 0) { results.ys[N] = workspace_.ys[N]; }
}

int OSCLQRSolver::check_termination() {
    int exit_flag = 0;
    if (prim_res_norm_ <= eps_prim_ && dual_res_norm_ <= eps_dual_) {
        exit_flag = 1; // converged
    }
    return exit_flag;
}

void OSCLQRSolver::update_rho(const LQRSettings& settings) {
    const scalar scale = std::sqrt((prim_res_norm_ * dual_res_norm_rel_) /
                                   (dual_res_norm_ * prim_res_norm_rel_));

    rho_estimate_sparse_ = std::max(settings.rho_min, std::min(settings.rho_max, rho_sparse_ * scale));
    
    if (rho_estimate_sparse_ > rho_sparse_ * settings.adaptive_rho_tolerance ||
        rho_estimate_sparse_ < rho_sparse_ / settings.adaptive_rho_tolerance) 
    {
        rho_sparse_ = rho_estimate_sparse_;
        for (int k = 0; k < model_.N + 1; ++k) {
            OSCLQRKernel::update_rho_step(model_.nodes[k],
                                          workspace_.rho_vecs[k], 
                                          workspace_.inv_rho_vecs[k], 
                                          rho_sparse_, settings.rho_min);
        }
    }
}

scalar OSCLQRSolver::compute_LQR_cost() {
    scalar cost = 0.0;
    for (int k = 0; k < model_.N; ++k) {
        const auto& kpoint = model_.nodes[k];
        const auto& w = workspace_.ws[k];
        cost += 0.5 * w.dot(kpoint.H * w) + kpoint.h.dot(w);
    }
    const auto& kpoint = model_.nodes.back();
    const auto& w = workspace_.ws.back();
    cost += 0.5 * w.dot(kpoint.H * w) + kpoint.h.dot(w);

    return cost;
}

void OSCLQRSolver::solve(const VectorXs& x0, const LQRSettings& settings, LQRResults& results) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);

    if (!is_rho_vec_init_) {
        init_rho(settings);
        is_rho_vec_init_ = true;
    }

    int can_check_termination = 0; 
    int can_update_rho = 0;

// #ifdef LQR_VERBOSE
    // Print header for iteration log
    std::cout << std::setw(4)  << "iter" 
              << std::setw(13) << "objective" 
              << std::setw(13) << "prim res" 
              << std::setw(13) << "dual res" 
              << std::setw(13) << "rho" << std::endl;   
// #endif
    for (int iter = 1; iter <= settings.max_iter; ++iter) {

        can_check_termination = (iter % settings.termination_check_interval == 0);
        
        lqr_solver_.update_problem_data(workspace_.ws, 
                                        workspace_.ys, 
                                        workspace_.zs, 
                                        workspace_.inv_rho_vecs, 
                                        settings.sigma);

        if (iter == 1 || can_update_rho) {
            lqr_solver_.backward(workspace_.rho_vecs);
        }
        else {
            lqr_solver_.backward_without_factorization(workspace_.rho_vecs);
        }

        lqr_solver_.forward(x0, workspace_.ws);

        update_dual_variables(settings.alpha);

        can_update_rho = (iter % settings.rho_update_interval == 0);

        if (iter == 1 || can_check_termination) {
            compute_residuals(settings);

// #ifdef LQR_VERBOSE
            // Calculate current objective value
            scalar current_cost = compute_LQR_cost();
            // Print iteration log
            std::cout << std::setw(4) << iter 
                      << std::scientific << std::setprecision(4)
                      << std::setw(13) << current_cost
                      << std::setw(13) << prim_res_norm_
                      << std::setw(13) << dual_res_norm_
                      << std::setw(13) << rho_sparse_ << std::endl;
// #endif
            if (check_termination() != 0) {
                results.optimal_LQR_cost = compute_LQR_cost();
// #ifdef LQR_VERBOSE
                std::cout << "\n"
                          << "optimal objective: " << results.optimal_LQR_cost << "\n" 
                          << "number of iterations: " << iter << "\n" << std::endl;
// #endif               
                update_results(results);
                return;
            }
        }

        if (can_update_rho) {
            if (!can_check_termination) {
                compute_residuals(settings);
            }
            update_rho(settings);
        }
    }
    std::cout << "Maximum iterations reached." << std::endl;
    update_results(results);
}

} // namespace lqr