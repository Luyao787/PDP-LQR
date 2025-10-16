#pragma once

#include <memory>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "lqr/typedefs.hpp"
#include "lqr/lqr_model.hpp"
#include "lqr/lqr_solver_base.hpp"
#include "lqr/clqr_kernel.hpp"
#include "lqr/settings.hpp"
#include "lqr/results.hpp"

namespace lqr {

template<class KernelData>
class CLQRSolver : public LQRSolverBase {
public:
    CLQRSolver(const LQRModel& model);
    void clear_workspace() { 
        for (auto& data : workspace_) data.set_zero(); 
        is_rho_vec_init_ = false;
    }
    
    void backward(scalar sigma) override;
    virtual void backward_without_factorization(scalar sigma);
    void forward(const VectorXs& x0) override;

    virtual void update_dual_variables(scalar alpha);
    
    void init_rho(const LQRSettings& settings);
    void update_rho(const LQRSettings& settings);

    void compute_residuals(const LQRSettings& settings);
    int check_termination();

    void update_results(LQRResults& results);
    scalar compute_LQR_cost();

    void solve(const VectorXs& x0, const LQRSettings& settings, LQRResults& results);

protected:
    const LQRModel& model_;
    scalar rho_sparse_; 
    scalar rho_estimate_sparse_;
    scalar prim_res_norm_; 
    scalar dual_res_norm_;
    scalar prim_res_norm_rel_; 
    scalar dual_res_norm_rel_;
    scalar eps_prim_; 
    scalar eps_dual_;    
    std::vector<KernelData> workspace_;
    bool is_rho_vec_init_ = false;
    
};

/* Definitions */

template<class KerD>
CLQRSolver<KerD>::CLQRSolver(const LQRModel& model)
    : model_(model)
{
    workspace_.reserve(model.N + 1);
    for (int k = 0; k < model.N; ++k) {
        workspace_.emplace_back(model.n, model.m, model.knotpoints[k].n_con);
    }
    workspace_.emplace_back(model.n, 0, model.knotpoints.back().n_con, true);

    prim_res_norm_     = 0.0;
    dual_res_norm_     = 0.0;
    prim_res_norm_rel_ = 0.0;
    dual_res_norm_rel_ = 0.0;
    eps_prim_          = 0.0;
    eps_dual_          = 0.0;

    rho_sparse_ = 0.1;
    rho_estimate_sparse_ = 0.1;
}

template<class KerD>
void CLQRSolver<KerD>::backward(scalar sigma) {
    CLQRKernel::terminal_step_with_factorization(model_.knotpoints.back(), workspace_.back(), sigma);
    for (int k = model_.N - 1; k >= 0; --k) {
        CLQRKernel::step_with_factorization(model_.knotpoints[k], workspace_[k], workspace_[k + 1], sigma);
    }
}

template<class KerD>
void CLQRSolver<KerD>::backward_without_factorization(scalar sigma) {
    CLQRKernel::terminal_step_without_factorization(model_.knotpoints.back(), workspace_.back(), sigma);
    for (int k = model_.N - 1; k >= 0; --k) {
        CLQRKernel::step_without_factorization(model_.knotpoints[k], workspace_[k], workspace_[k + 1], sigma);
    }
}

template<class KerD>
void CLQRSolver<KerD>::forward(const VectorXs& x0) {
    workspace_[0].w.tail(model_.n) = x0;
    for (int k = 0; k < model_.N; ++k) {
        CLQRKernel::forward_step(model_.knotpoints[k], workspace_[k], workspace_[k + 1]);
    }   
}

template<class KerD>
void CLQRSolver<KerD>::init_rho(const LQRSettings& settings) {
    rho_sparse_ = settings.rho;
    rho_estimate_sparse_ = settings.rho;
    for (int k = 0; k < model_.N + 1; ++k) {
        workspace_[k].rho_vec.setConstant(rho_sparse_);
        workspace_[k].inv_rho_vec.setConstant(1.0 / rho_sparse_);
    }
}

template<class KerD>
void CLQRSolver<KerD>::update_dual_variables(scalar alpha) {
    for (int k = 0; k < model_.N + 1; ++k) {
        auto& data = workspace_[k];
        auto& kpoint = model_.knotpoints[k];
        if (kpoint.n_con > 0) {
            data.Dw_tmp.noalias()   = kpoint.D_con * data.w;

            data.z_prev = data.z; // store previous z

            data.z_tilde.noalias()  = alpha * data.Dw_tmp;
            data.z_tilde.noalias() += (1. - alpha) * data.z;

            data.y_tmp = data.inv_rho_vec.cwiseProduct(data.y);
            data.z     = data.z_tilde + data.y_tmp;
            data.z     = data.z.cwiseMax(kpoint.e_lb).cwiseMin(kpoint.e_ub);

            data.y    += data.rho_vec.cwiseProduct(data.z_tilde - data.z);
        }
    }
}

template<class KerD>
void CLQRSolver<KerD>::compute_residuals(const LQRSettings& settings) {

    prim_res_norm_     = 0.0;
    dual_res_norm_     = 0.0;
    prim_res_norm_rel_ = 0.0;
    dual_res_norm_rel_ = 0.0;

    for (int k = 0; k < model_.N; ++k) {
        auto& data = workspace_[k];
        auto& data_next = workspace_[k + 1];
        auto& kpoint = model_.knotpoints[k];
        CLQRKernel::residual_step(kpoint, data, data_next, prim_res_norm_, dual_res_norm_, prim_res_norm_rel_, dual_res_norm_rel_);
    }
    {
        auto& data = workspace_.back();
        auto& kpoint = model_.knotpoints.back();
        CLQRKernel::residual_terminal_step(kpoint, data, prim_res_norm_, dual_res_norm_, prim_res_norm_rel_, dual_res_norm_rel_);
    }

    eps_prim_ = settings.eps_abs + prim_res_norm_rel_ * settings.eps_rel;
    eps_dual_ = settings.eps_abs + dual_res_norm_rel_ * settings.eps_rel;
}

template<class KerD>
void CLQRSolver<KerD>::update_results(LQRResults& results) {
    int nx = model_.n;
    int nu = model_.m;
    int N  = model_.N;
    for (int k = 0; k < model_.N; ++k) {
        results.xs[k] = workspace_[k].w.tail(nx);
        results.us[k] = workspace_[k].w.head(nu);
        results.lambdas[k] = workspace_[k].lambda;
        if (model_.knotpoints[k].n_con > 0) { results.ys[k] = workspace_[k].y; }
    }
    results.xs[N] = workspace_[N].w;
    results.lambdas[N] = workspace_[N].lambda;
    if (model_.knotpoints[N].n_con > 0) { results.ys[N] = workspace_[N].y; }
}

template<class KerD>
int CLQRSolver<KerD>::check_termination() {
    int exit_flag = 0;
    if (prim_res_norm_ <= eps_prim_ && dual_res_norm_ <= eps_dual_) {
        exit_flag = 1; // converged
    }
    return exit_flag;
}

template<class KerD>
void CLQRSolver<KerD>::update_rho(const LQRSettings& settings) {
    const scalar scale = std::sqrt((prim_res_norm_ * dual_res_norm_rel_) /
                                   (dual_res_norm_ * prim_res_norm_rel_));

    rho_estimate_sparse_ = std::max(settings.rho_min, std::min(settings.rho_max, rho_sparse_ * scale));
    
    if (rho_estimate_sparse_ > rho_sparse_ * settings.adaptive_rho_tolerance ||
        rho_estimate_sparse_ < rho_sparse_ / settings.adaptive_rho_tolerance) 
    {
        rho_sparse_ = rho_estimate_sparse_;
        
        for (int k = 0; k < model_.N + 1; ++k) {
            CLQRKernel::update_rho_step(model_.knotpoints[k], workspace_[k], rho_sparse_, settings.rho_min);
        }
    }
}

template<class KerD>
scalar CLQRSolver<KerD>::compute_LQR_cost() {
    scalar cost = 0.0;
    for (int k = 0; k < model_.N; ++k) {
        auto& kpoint = model_.knotpoints[k];
        auto& data = workspace_[k];
        cost += kpoint.h.dot(data.w); 
        data.Hw_tmp.noalias() = kpoint.H * data.w;
        cost += 0.5 * data.w.dot(data.Hw_tmp);
    }
    {
        auto& kpoint = model_.knotpoints[model_.N];
        auto& data = workspace_[model_.N];
        cost += kpoint.h.dot(data.w);
        data.Hw_tmp.noalias() = kpoint.H * data.w;
        cost += 0.5 * data.w.dot(data.Hw_tmp);
    }

    return cost;
}

template<class KerD>
void CLQRSolver<KerD>::solve(const VectorXs& x0, const LQRSettings& settings, LQRResults& results) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);
   
    if (!is_rho_vec_init_) {
        init_rho(settings);
        is_rho_vec_init_ = true;
    }

    int can_check_termination = 0; 
    int can_update_rho = 0;

#ifdef LQR_VERBOSE
    // Print header for iteration log
    std::cout << std::setw(4)  << "iter" 
              << std::setw(13) << "objective" 
              << std::setw(13) << "prim res" 
              << std::setw(13) << "dual res" 
              << std::setw(13) << "rho" << std::endl;   
#endif
    for (int iter = 1; iter <= settings.max_iter; ++iter) {

        can_check_termination = (iter % settings.termination_check_interval == 0);
        
        // auto tic = omp_get_wtime();
        if (iter == 1 || can_update_rho) {
            backward(settings.sigma);
        }
        else {
            backward_without_factorization(settings.sigma);
        }
        // auto toc = omp_get_wtime();
        // std::cout << "Time for backward pass: " << (toc - tic) * 1e3 << " ms" << std::endl;

        // tic = omp_get_wtime();
        forward(x0);
        // toc = omp_get_wtime();
        // std::cout << "Time for forward pass: " << (toc - tic) * 1e3 << " ms" << std::endl;

        // tic = omp_get_wtime();
        update_dual_variables(settings.alpha);
        // toc = omp_get_wtime();
        // std::cout << "Time for dual update: " << (toc - tic) * 1e3 << " ms" << std::endl;
        // std::cout << "----------------------------------------" << std::endl;

        can_update_rho = (iter % settings.rho_update_interval == 0);
    
        if (iter == 1 || can_check_termination) {
            compute_residuals(settings);
        
#ifdef LQR_VERBOSE
            // Calculate current objective value
            scalar current_cost = compute_LQR_cost();
            // Print iteration log
            std::cout << std::setw(4) << iter 
                      << std::scientific << std::setprecision(4)
                      << std::setw(13) << current_cost
                      << std::setw(13) << prim_res_norm_
                      << std::setw(13) << dual_res_norm_
                      << std::setw(13) << rho_sparse_ << std::endl;
#endif
            if (check_termination() != 0) {
                results.optimal_LQR_cost = compute_LQR_cost();
#ifdef LQR_VERBOSE
                std::cout << "\n"
                          << "optimal objective: " << results.optimal_LQR_cost << "\n" 
                          << "number of iterations: " << iter << "\n" << std::endl;
#endif               
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

// using SerialCLQRSolver = CLQRSolver<CLQRKernelData>;

} // namespace lqr