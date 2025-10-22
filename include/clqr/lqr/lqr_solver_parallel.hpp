#pragma once 

#include <memory>
#include "clqr/typedefs.hpp"
#include "clqr/lqr_model.hpp"
#include "clqr/lqr/lqr_kernel_parallel.hpp"
#include "clqr/lqr/condensed_system.hpp"

namespace lqr {

class LQRParallelSolver {
public:
    LQRParallelSolver() = default;
    LQRParallelSolver(const LQRModel& model, int num_segments, bool load_balancing=true);
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
    void reduction(int thread_id, const std::vector<VectorXs>& rho_vecs);
    void reduction_without_factorization(int thread_id, const std::vector<VectorXs>& rho_vecs);
    void consensus();
    void consensus_without_factorization();
    
    void forward(const VectorXs& x0, std::vector<VectorXs>& ws);

private:
    const LQRModel& model_;  
    std::vector<ParallelLQRKernelData> workspace_;
    int num_segments_;
    std::vector<int> Nseg_;
    std::vector<int> idx_start_;
    // std::vector<CondensedLQRKernelData> condensed_workspace_;
    std::vector<ParallelLQRKernelData> segment_terminal_workspace_;

    std::unique_ptr<CondensedSystemLUSolver> condensed_system_solver_;
    // std::unique_ptr<CondensedSystemCholeskySolver> condensed_system_solver_;

};

LQRParallelSolver::LQRParallelSolver(const LQRModel& model, int num_segments, bool load_balancing)
    : model_(model), num_segments_(num_segments)
{
    workspace_.reserve(model.N + 1);
    for (int k = 0; k < model.N; ++k) {
        workspace_.emplace_back(model.n, model.m, model.nodes[k].n_con);
    }
    workspace_.emplace_back(model.n, 0, model.nodes.back().n_con, true);

    Nseg_.resize(num_segments_);
    idx_start_.resize(num_segments_);
    idx_start_[0] = 0;  // Initialize first index

    scalar alpha = 1.55;
    double scale = load_balancing ? alpha : 1.0;

     for (int i = 0; i < num_segments_ - 1; ++i) {
        Nseg_[i] = int(model.N / (scale + num_segments_ - 1));
        idx_start_[i + 1] = idx_start_[i] + Nseg_[i];
    }
    Nseg_[num_segments_ - 1] = model.N - idx_start_[num_segments_ - 1];

    // condensed_workspace_.reserve(num_segments_);
    // for (int i = 0; i < num_segments_; ++i) {
    //     condensed_workspace_.emplace_back(model.n);
    // }
    condensed_system_solver_ = std::make_unique<CondensedSystemLUSolver>(model.n, num_segments_);
    // condensed_system_solver_ = std::make_unique<CondensedSystemCholeskySolver>(model.n, num_segments_);

    segment_terminal_workspace_.reserve(num_segments_);
    for (int i = 0; i < num_segments_; ++i) {
        const auto& kpoint = model.nodes[ idx_start_[i] + Nseg_[i] ];
        segment_terminal_workspace_.emplace_back(kpoint.n, kpoint.m, kpoint.n_con, kpoint.is_terminal);
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

void LQRParallelSolver::backward(const std::vector<VectorXs>& rho_vecs) {
    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        reduction(tid, rho_vecs);
    }
    condensed_system_solver_->backward();
}

void LQRParallelSolver::backward_without_factorization(const std::vector<VectorXs>& rho_vecs) {
    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        reduction_without_factorization(tid, rho_vecs);
    }
}

void LQRParallelSolver::reduction(int thread_id, const std::vector<VectorXs>& rho_vecs) {
    int N0 = idx_start_[thread_id];
    int N1 = N0 + Nseg_[thread_id];
    bool is_last_segment = (thread_id == num_segments_ - 1);

    ParallelLQRKernel::terminal_step_with_factorization(model_.nodes[N1], rho_vecs[N1], workspace_[N1], is_last_segment);
    for (int k = N1 - 1; k >= N0; --k) {
        ParallelLQRKernel::step_with_factorization(
            model_.nodes[k], rho_vecs[k], workspace_[k + 1], workspace_[k], is_last_segment);
    } 
    int nx = model_.n;
    ConstMatrixRef Lxx_N0 = workspace_[N0].L.bottomRightCorner(nx, nx);   
    // condensed_workspace_[thread_id].Q = Lxx_N0 * Lxx_N0.transpose();
    // condensed_workspace_[thread_id].q = workspace_[N0].lp.tail(nx);
    // condensed_workspace_[thread_id].A = workspace_[N0].F;
    // condensed_workspace_[thread_id].C = workspace_[N0].C;
    // condensed_workspace_[thread_id].c = workspace_[N0].f;
    // condensed_workspace_[thread_id].P = Lxx_N0 * Lxx_N0.transpose();
    // condensed_workspace_[thread_id].p = workspace_[N0].lp.tail(nx);
    // condensed_workspace_[thread_id].A = workspace_[N0].F;
    // condensed_workspace_[thread_id].C = workspace_[N0].C;
    // condensed_workspace_[thread_id].c = workspace_[N0].f;

    condensed_system_solver_->update_segment_data(
        Lxx_N0,
        workspace_[N0].F,
        workspace_[N0].C,
        workspace_[N0].lp.tail(nx),
        workspace_[N0].f, thread_id);
}

void LQRParallelSolver::reduction_without_factorization(int thread_id, const std::vector<VectorXs>& rho_vecs) {
    int N0 = idx_start_[thread_id];
    int N1 = N0 + Nseg_[thread_id];
    bool is_last_segment = (thread_id == num_segments_ - 1);

    segment_terminal_workspace_[thread_id] = workspace_[N1];
    ParallelLQRKernel::terminal_step_without_factorization(
        model_.nodes[N1], rho_vecs[N1], segment_terminal_workspace_[thread_id], is_last_segment);
    if (N1 - 1 >= N0) {
        ParallelLQRKernel::step_without_factorization(
            model_.nodes[N1 - 1], rho_vecs[N1 - 1], segment_terminal_workspace_[thread_id], workspace_[N1 - 1], is_last_segment);
        for (int k = N1 - 2; k >= N0; --k) {
            ParallelLQRKernel::step_without_factorization(
                model_.nodes[k], rho_vecs[k], workspace_[k + 1], workspace_[k], is_last_segment);
        }
    }
    int nx = model_.n;
    // condensed_workspace_[thread_id].q = workspace_[N0].lp.tail(nx);
    // condensed_workspace_[thread_id].c = workspace_[N0].f;
    // condensed_workspace_[thread_id].p = workspace_[N0].lp.tail(nx);
    // condensed_workspace_[thread_id].c = workspace_[N0].f;
    condensed_system_solver_->update_segment_data(
        workspace_[N0].lp.tail(nx),
        workspace_[N0].f, thread_id);
}

// void LQRParallelSolver::consensus() {
//     condensed_workspace_[num_segments_-1].P = condensed_workspace_[num_segments_-1].Q;
//     condensed_workspace_[num_segments_-1].p = condensed_workspace_[num_segments_-1].q;

//     for (int i = num_segments_ - 2; i >= 0; --i) {
//         ConstMatrixRef P_next = condensed_workspace_[i+1].P;
//         ConstVectorRef p_next = condensed_workspace_[i+1].p;

//         ConstMatrixRef A_i = condensed_workspace_[i].A;
//         ConstMatrixRef C_i = condensed_workspace_[i].C;
//         ConstMatrixRef Q_i = condensed_workspace_[i].Q;
//         ConstVectorRef c_i = condensed_workspace_[i].c;
//         ConstVectorRef q_i = condensed_workspace_[i].q;
//         MatrixRef P_i      = condensed_workspace_[i].P;
//         VectorRef p_i      = condensed_workspace_[i].p;
//         MatrixRef PC_i     = condensed_workspace_[i].PC;
//         MatrixRef PA_i     = condensed_workspace_[i].PA;
//         MatrixRef D_i      = condensed_workspace_[i].D;
//         VectorRef Pc       = condensed_workspace_[i].Pc;

//         PC_i.noalias() = C_i * P_next;
//         PC_i.diagonal().array() += 1.0;
//         PA_i.noalias() = P_next * A_i; 

//         condensed_workspace_[i].lu_fact.compute(PC_i);
//         D_i = condensed_workspace_[i].lu_fact.solve(A_i);
//         P_i = Q_i;
//         P_i.noalias() += D_i.transpose() * PA_i;

//         Pc = p_next;
//         Pc.noalias() += P_next * c_i;
//         p_i = q_i;
//         p_i.noalias() += D_i.transpose() * Pc;
//     }
// }

// void LQRParallelSolver::consensus_without_factorization() {
//     condensed_workspace_[num_segments_-1].p = condensed_workspace_[num_segments_-1].q;

//     for (int i = num_segments_ - 2; i >= 0; --i) {
//         ConstMatrixRef P_next = condensed_workspace_[i+1].P;
//         ConstVectorRef p_next = condensed_workspace_[i+1].p;

//         ConstVectorRef c_i = condensed_workspace_[i].c;
//         ConstVectorRef q_i = condensed_workspace_[i].q;
//         VectorRef p_i      = condensed_workspace_[i].p;
//         MatrixRef D_i      = condensed_workspace_[i].D;
//         VectorRef Pc       = condensed_workspace_[i].Pc;

//         Pc = p_next;
//         Pc.noalias() += P_next * c_i;
//         p_i = q_i;
//         p_i.noalias() += D_i.transpose() * Pc;
//     }    
// }

// void LQRParallelSolver::consensus() {

//     for (int i = num_segments_ - 2; i >= 0; --i) {
//         ConstMatrixRef P_next = condensed_workspace_[i+1].P;
//         ConstVectorRef p_next = condensed_workspace_[i+1].p;

//         ConstMatrixRef A_i = condensed_workspace_[i].A;
//         ConstMatrixRef C_i = condensed_workspace_[i].C;
//         ConstVectorRef c_i = condensed_workspace_[i].c;
//         MatrixRef P_i      = condensed_workspace_[i].P;
//         VectorRef p_i      = condensed_workspace_[i].p;
//         MatrixRef PC_i  = condensed_workspace_[i].PC;
//         MatrixRef PA_i     = condensed_workspace_[i].PA;
//         MatrixRef D_i      = condensed_workspace_[i].D;
//         VectorRef c_bar_i       = condensed_workspace_[i].c_bar;

//         PC_i.noalias() = C_i * P_next;
//         PC_i.diagonal().array() += 1.0;
//         PA_i.noalias() = P_next * A_i; 

//         condensed_workspace_[i].lu_fact.compute(PC_i);
//         D_i = condensed_workspace_[i].lu_fact.solve(A_i);
//         // P_i = Q_i;
//         P_i.noalias() += D_i.transpose() * PA_i;

//         // Pc = p_next;
//         // Pc.noalias() += P_next * c_i;
//         // p_i = q_i;
//         // p_i.noalias() += D_i.transpose() * Pc;
//         c_bar_i = p_next;
//         c_bar_i.noalias() += P_next * c_i;
//         p_i.noalias() += D_i.transpose() * c_bar_i;        
   
//     }
// }

// void LQRParallelSolver::consensus_without_factorization() {
//     for (int i = num_segments_ - 2; i >= 0; --i) {
//         ConstMatrixRef P_next = condensed_workspace_[i+1].P;
//         ConstVectorRef p_next = condensed_workspace_[i+1].p;

//         ConstVectorRef c_i = condensed_workspace_[i].c;
//         // ConstVectorRef q_i = condensed_workspace_[i].q;
//         VectorRef p_i      = condensed_workspace_[i].p;
//         MatrixRef D_i      = condensed_workspace_[i].D;
//         VectorRef c_bar_i       = condensed_workspace_[i].c_bar;

//         // Pc = p_next;
//         // Pc.noalias() += P_next * c_i;
//         // p_i = q_i;
//         // p_i.noalias() += D_i.transpose() * Pc;
//         c_bar_i = p_next;
//         c_bar_i.noalias() += P_next * c_i;
//         p_i.noalias() += D_i.transpose() * c_bar_i;
//     }    
// }

void LQRParallelSolver::forward(const VectorXs& x0, std::vector<VectorXs>& ws) {
    // condensed_workspace_[0].xhat = x0;
    // for (int i = 0; i < num_segments_ - 1; ++i) {
    //     ConstMatrixRef P_next = condensed_workspace_[i + 1].P;
    //     ConstVectorRef p_next = condensed_workspace_[i + 1].p;
    //     ConstMatrixRef A_i    = condensed_workspace_[i].A;
    //     ConstMatrixRef C_i    = condensed_workspace_[i].C;
    //     VectorRef c_i         = condensed_workspace_[i].c;
    //     // VectorRef e_i         = condensed_workspace_[i].e;
        
    //     auto& xhat      = condensed_workspace_[i].xhat;
    //     auto& uhat      = condensed_workspace_[i].uhat;
    //     auto& xhat_next = condensed_workspace_[i + 1].xhat;
    //     // e_i = c_i;
    //     // e_i.noalias() += A_i * xhat;
    //     // e_i.noalias() -= C_i * p_next;
    //     c_i.noalias() += A_i * xhat;
    //     c_i.noalias() -= C_i * p_next;

    //     // xhat_next = condensed_workspace_[i].lu_fact.solve(e_i);
    //     xhat_next = condensed_workspace_[i].lu_fact.solve(c_i);
    //     uhat = p_next;
    //     uhat.noalias() += P_next * xhat_next;
    // }

    condensed_system_solver_->forward(x0);

    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        int N0 = idx_start_[tid];
        int N1 = N0 + Nseg_[tid];
        bool is_last_segment = (tid == num_segments_ - 1);
    
        if (is_last_segment) {
            workspace_[N1].lp = segment_terminal_workspace_[tid].lp; // TODO improve efficiency
        }
        ws[N0].tail(model_.n) = condensed_system_solver_->get_xhat(tid);
        const auto& uhat = condensed_system_solver_->get_uhat(tid);
        for (int k = N0; k < N1; ++k) {
            ParallelLQRKernel::forward_step(
                model_.nodes[k],
                uhat,
                workspace_[k], 
                ws[k],
                ws[k + 1],
                is_last_segment);
        }
        if (tid < num_segments_ - 1) {
            // workspace_[N1].lambda = condensed_workspace_[tid].uhat; // TODO 
        }
    }   
}

} // namespace lqr