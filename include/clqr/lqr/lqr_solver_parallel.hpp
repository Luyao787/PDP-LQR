#pragma once 

#include <iostream>
#include <memory>
#include <chrono>
#include "clqr/typedefs.hpp"
#include "clqr/lqr_model.hpp"
#include "clqr/lqr/lqr_kernel_parallel.hpp"
#include "clqr/lqr/condensed_system.hpp"
// #include <tracy/Tracy.hpp>

namespace lqr {

enum class CondensedSystemSolverType {
    LU,
    CHOLESKY
};

class LQRParallelSolver {
public:
    LQRParallelSolver() = default;
    LQRParallelSolver(const LQRModel& model, 
                      int num_segments, 
                      bool load_balancing=true, 
                      CondensedSystemSolverType solver_type = CondensedSystemSolverType::CHOLESKY);
    void clear_workspace() {
        for (auto& segment : workspace_) {
            for (auto& data : segment.data) {
                data.set_zero();
            }
        } 
    }
    void update_problem_data(const std::vector<VectorXs>& ws, 
                             const std::vector<VectorXs>& ys, 
                             const std::vector<VectorXs>& zs,
                             const std::vector<VectorXs>& inv_rho_vecs,
                             const scalar sigma);

    void backward(const std::vector<VectorXs>& rho_vecs);
    void backward_without_factorization(const std::vector<VectorXs>& rho_vecs);
    // void reduction(int thread_id, const std::vector<VectorXs>& rho_vecs);
    void reduction(const std::vector<VectorXs>& rho_vecs);
    void reduction_per_thread(int thread_id, const std::vector<VectorXs>& rho_vecs);

    void reduction_without_factorization(int thread_id, const std::vector<VectorXs>& rho_vecs);
    void consensus();
    void consensus_without_factorization();
    
    void forward(const VectorXs& x0, std::vector<VectorXs>& ws);

private:
    const LQRModel& model_;  
    int num_segments_;
    std::unique_ptr<CondensedSystemSolverBase> condensed_system_solver_;
    struct ThreadWorkspace {
        std::vector<ParallelLQRKernelData> data;
        int idx_start;
        int Nseg;
    };
    std::vector<ThreadWorkspace> workspace_;

};

LQRParallelSolver::LQRParallelSolver(const LQRModel& model, 
                                     int num_segments, 
                                     bool load_balancing,  
                                     CondensedSystemSolverType solver_type)
    : model_(model), num_segments_(num_segments)
{
    scalar alpha = 1.55; // TODO: remove magic number
    double scale = load_balancing ? alpha : 1.0;

    workspace_.resize(num_segments_);
    for (int i = 0; i < num_segments_; ++i) {
        int idx_start, Nseg;
        idx_start = (i == 0) ? 0 : workspace_[i - 1].idx_start + workspace_[i - 1].Nseg;
        Nseg = (i < num_segments_ - 1) ? 
               int(model.N / (scale + num_segments_ - 1)) : model.N - idx_start;
        workspace_[i].idx_start = idx_start;
        workspace_[i].Nseg = Nseg;
        workspace_[i].data.reserve(Nseg + 1);

        for (int k = 0; k <= Nseg; ++k) {
            const auto& node = model_.nodes[idx_start + k];
            workspace_[i].data.emplace_back(
                node.n, node.m, node.n_con, node.is_terminal);
        }
    }

    switch (solver_type)
    {
    case CondensedSystemSolverType::LU:
        condensed_system_solver_ = std::make_unique<CondensedSystemLUSolver>(model.n, num_segments_);
        break;
    case CondensedSystemSolverType::CHOLESKY:
        condensed_system_solver_ = std::make_unique<CondensedSystemCholeskySolver>(model.n, num_segments_);
        break;
    default:
        throw std::runtime_error("Unsupported CondensedSystemSolverType");
    }

    #pragma omp parallel num_threads(num_segments_)
    {
        /* Set thread affinity to CPU core */ 
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid, &cpuset);
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) < 0) {
            printf("Error setting thread affinity for thread %d\n", tid);
        }
    }
}

void LQRParallelSolver::update_problem_data(const std::vector<VectorXs>& ws, 
                                            const std::vector<VectorXs>& ys, 
                                            const std::vector<VectorXs>& zs,
                                            const std::vector<VectorXs>& inv_rho_vecs,
                                            const scalar sigma) {
    #pragma omp parallel num_threads(num_segments_) 
    {
        const int tid = omp_get_thread_num();
        const int Nseg = workspace_[tid].Nseg;
        const int N0 = workspace_[tid].idx_start;

        for (int k = 0; k < Nseg + 1; ++k) {
            int idx_global = N0 + k;
            auto& kpoint = model_.nodes[idx_global];
            workspace_[tid].data[k].H = kpoint.H;
            workspace_[tid].data[k].H.diagonal().array() += sigma;
            workspace_[tid].data[k].h = kpoint.h;
            workspace_[tid].data[k].h.noalias() -= sigma * ws[idx_global];

            if (kpoint.n_con > 0) {
                workspace_[tid].data[k].g = zs[idx_global];
                workspace_[tid].data[k].g.noalias() -= inv_rho_vecs[idx_global].cwiseProduct(ys[idx_global]);
            }
        }
    }
}

void LQRParallelSolver::backward(const std::vector<VectorXs>& rho_vecs) { 
    // ZoneScoped;
    reduction(rho_vecs);
    condensed_system_solver_->backward();
}

void LQRParallelSolver::backward_without_factorization(const std::vector<VectorXs>& rho_vecs) {
    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        reduction_without_factorization(tid, rho_vecs);
    }
}

void LQRParallelSolver::reduction(const std::vector<VectorXs>& rho_vecs) {
    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        reduction_per_thread(tid, rho_vecs);
    }
}

void LQRParallelSolver::reduction_per_thread(int thread_id, const std::vector<VectorXs>& rho_vecs) {
    const int N0 = workspace_[thread_id].idx_start;
    const int Nseg = workspace_[thread_id].Nseg;
    const int N1 = N0 + Nseg;
    bool is_last_segment = (thread_id == num_segments_ - 1);
    
    ParallelLQRKernel::terminal_step_with_factorization(
        model_.nodes[N1], rho_vecs[N1], workspace_[thread_id].data.back(), is_last_segment);    
    for (int k = N1 - 1; k >= N0; --k) {
        ParallelLQRKernel::step_with_factorization(
            model_.nodes[k], 
            rho_vecs[k], 
            workspace_[thread_id].data[k - N0 + 1], 
            workspace_[thread_id].data[k - N0], 
            is_last_segment);
    }
    int nx = model_.n;
    ConstMatrixRef Lxx_N0 = workspace_[thread_id].data[0].L.bottomRightCorner(nx, nx);
    condensed_system_solver_->update_segment_data(
        Lxx_N0,
        workspace_[thread_id].data[0].F,
        workspace_[thread_id].data[0].C,
        workspace_[thread_id].data[0].lp.tail(nx),
        workspace_[thread_id].data[0].f, thread_id);
}

void LQRParallelSolver::reduction_without_factorization(int thread_id, const std::vector<VectorXs>& rho_vecs) {
    const int N0 = workspace_[thread_id].idx_start;
    const int Nseg = workspace_[thread_id].Nseg;
    const int N1 = N0 + Nseg;
    bool is_last_segment = (thread_id == num_segments_ - 1);

    ParallelLQRKernel::terminal_step_without_factorization(
        model_.nodes[N1], rho_vecs[N1], workspace_[thread_id].data.back(), is_last_segment);    
    for (int k = N1 - 1; k >= N0; --k) {
        ParallelLQRKernel::step_without_factorization(
            model_.nodes[k], 
            rho_vecs[k], 
            workspace_[thread_id].data[k - N0 + 1], 
            workspace_[thread_id].data[k - N0], 
            is_last_segment);
    }
    const int nx = model_.n;
    condensed_system_solver_->update_segment_data(
        workspace_[thread_id].data[0].lp.tail(nx), 
        workspace_[thread_id].data[0].f, 
        thread_id);
}

void LQRParallelSolver::forward(const VectorXs& x0, std::vector<VectorXs>& ws) {
    
    condensed_system_solver_->forward(x0);

    #pragma omp parallel num_threads(num_segments_) 
    {
        const int tid = omp_get_thread_num();
        const int N0 = workspace_[tid].idx_start;
        const int N1 = N0 + workspace_[tid].Nseg;
        bool is_last_segment = (tid == num_segments_ - 1);
    
        ws[N0].tail(model_.n) = condensed_system_solver_->get_xhat(tid);
        const auto& uhat = condensed_system_solver_->get_uhat(tid);
        
        for (int k = N0; k < N1; ++k) {
            ParallelLQRKernel::forward_step(
                model_.nodes[k],
                uhat,
                workspace_[tid].data[k - N0],
                ws[k],
                ws[k + 1],
                is_last_segment, 
                (k < N1 - 1));
        }
    }   
}

} // namespace lqr