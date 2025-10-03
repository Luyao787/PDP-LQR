#pragma once

#include <memory>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <sched.h>
#include "lqr/typedefs.hpp"
#include "lqr/lqr_model.hpp"
#include "lqr/lqr_solver_base.hpp"
#include "lqr/clqr_kernel.hpp"
#include "lqr/settings.hpp"
#include "lqr/results.hpp"
#include "lqr/clqr_solver.hpp"
#include "lqr/clqr_parallel_kernel.hpp"

namespace lqr {

template<class KernelData>
class CLQRParallelSolver : public CLQRSolver<KernelData> {
public:
    CLQRParallelSolver(const LQRModel& model, int num_segments, bool load_balancing=true);

    void backward(scalar sigma);
    void backward_without_factorization(scalar sigma) override;
    void forward(const VectorXs& x0);
    void reduction(int thread_id, scalar sigma);
    void reduction_without_factorization(int thread_id, scalar sigma);
    void consensus();
    void consensus_without_factorization();
    void update_dual_variables(scalar alpha) override;

    // To be deleted
    void reduction(int N0, int N1, bool is_last_segment, scalar sigma);

private:
    int num_segments_;
    std::vector<int> Nseg_;
    std::vector<int> idx_start_;
    std::vector<ParallelCLQRCondensedKernelData> condensed_workspace_;
    std::vector<ParallelCLQRKernelData> terminal_workspace_;
};

template<class KerD>
CLQRParallelSolver<KerD>::CLQRParallelSolver(const LQRModel& model, int num_segments, bool load_balancing)
    : CLQRSolver<KerD>(model), num_segments_(num_segments)
{
    Nseg_.resize(num_segments_);
    idx_start_.resize(num_segments_);
    idx_start_[0] = 0;  // Initialize first index

    // scalar alpha = (model.N <= 30) ? 1.6 : 1.0;
    scalar alpha = 1.55;
    double scale = load_balancing ? alpha : 1.0;

    for (int i = 0; i < num_segments_ - 1; ++i) {
        Nseg_[i] = int(model.N / (scale + num_segments_ - 1));
        idx_start_[i + 1] = idx_start_[i] + Nseg_[i];
    }
    Nseg_[num_segments_ - 1] = model.N - idx_start_[num_segments_ - 1];

    condensed_workspace_.reserve(num_segments_);
    for (int i = 0; i < num_segments_; ++i) {
        condensed_workspace_.emplace_back(model.n);
    }

    terminal_workspace_.reserve(num_segments_);
    for (int i = 0; i < num_segments_; ++i) {
        const auto& kpoint = model.knotpoints[ idx_start_[i] + Nseg_[i] ];
        terminal_workspace_.emplace_back(kpoint.n, kpoint.m, kpoint.n_con, kpoint.is_terminal);
    }

    bool verbose = true;
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
        // if (verbose) {
            // printf("WARMUP: Thread %d running on CPU %d\n", tid, sched_getcpu());
        // }
    }
    // if (verbose) {
    //     printf("ParLQR setup!\n");
    //     printf("num_segments: %d\n", num_segments_);
    //     for (int i = 0; i < num_segments_; ++i) {
    //         printf("Nseg[%d]: %d\n", i, Nseg_[i]);
    //     }
    // }
}

template<class KerD>
void CLQRParallelSolver<KerD>::backward(scalar sigma) {

    // Eigen::setNbThreads(1); // disable internal parallelization in Eigen

    #pragma omp parallel num_threads(num_segments_) 
    {
        // auto tic = omp_get_wtime();
        int tid = omp_get_thread_num();
        reduction(tid, sigma);
        // auto toc = omp_get_wtime();
        // #pragma omp critical
        // {
            // std::cout << "Thread id: " << tid << std::endl;
            // std::cout << "Time for reduction: " << (toc - tic) * 1e3 << " ms" << std::endl;
        // }
    }
    // #pragma omp critical
    // {
        // auto tic = omp_get_wtime();
        // consensus();
        // auto toc = omp_get_wtime();
        // std::cout << "Time for consensus: " << (toc - tic) * 1e3 << " ms" << std::endl;
    // }
    consensus();
}

template<class KerD>
void CLQRParallelSolver<KerD>::backward_without_factorization(scalar sigma) {
    // Eigen::setNbThreads(1); // disable internal parallelization in Eigen

    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        reduction_without_factorization(tid, sigma);
    }
    consensus_without_factorization();
}

template<class KerD>
void CLQRParallelSolver<KerD>::forward(const VectorXs& x0) {
    condensed_workspace_[0].xhat = x0;

    for (int i = 0; i < num_segments_ - 1; ++i) {
        ConstMatrixRef P_next = condensed_workspace_[i + 1].P;
        ConstVectorRef p_next = condensed_workspace_[i + 1].p;
        ConstMatrixRef A_i    = condensed_workspace_[i].A;
        ConstMatrixRef C_i    = condensed_workspace_[i].C;
        VectorRef c_i         = condensed_workspace_[i].c;
        VectorRef e_i         = condensed_workspace_[i].e;
        
        auto& xhat      = condensed_workspace_[i].xhat;
        auto& uhat      = condensed_workspace_[i].uhat;
        auto& xhat_next = condensed_workspace_[i + 1].xhat;
        e_i = c_i;
        e_i.noalias() += A_i * xhat;
        e_i.noalias() -= C_i * p_next;

        xhat_next = condensed_workspace_[i].lu_fact.solve(e_i);
        uhat = p_next;
        uhat.noalias() += P_next * xhat_next;
    }

    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        int N0 = idx_start_[tid];
        int N1 = N0 + Nseg_[tid];
        bool is_last_segment = (tid == num_segments_ - 1);
    
        if (is_last_segment) {
            this->workspace_[N1].lp = terminal_workspace_[tid].lp; // TODO improve efficiency
        }
        this->workspace_[N0].w.tail(this->model_.n) = condensed_workspace_[tid].xhat;
        for (int k = N0; k < N1; ++k) {
            ParallelCLQRKernel::forward_step(this->model_.knotpoints[k], 
                this->workspace_[k], this->workspace_[k + 1], condensed_workspace_[tid], is_last_segment);
        }
        if (tid < num_segments_ - 1) {
            this->workspace_[N1].lambda = condensed_workspace_[tid].uhat;
        }
    }   
}

template<class KerD>
void CLQRParallelSolver<KerD>::reduction(int thread_id, scalar sigma) {
    int N0 = idx_start_[thread_id];
    int N1 = N0 + Nseg_[thread_id];
    bool is_last_segment = (thread_id == num_segments_ - 1);

    ParallelCLQRKernel::terminal_step_with_factorization(
        this->model_.knotpoints[N1], this->workspace_[N1], sigma, is_last_segment);
    for (int k = N1 - 1; k >= N0; --k) {
        ParallelCLQRKernel::step_with_factorization(
            this->model_.knotpoints[k], this->workspace_[k], this->workspace_[k + 1], sigma, is_last_segment);
    }
    
    int nx = this->model_.n;
    ConstMatrixRef Lxx_N0 = this->workspace_[N0].L.bottomRightCorner(nx, nx);   
    condensed_workspace_[thread_id].Q = Lxx_N0 * Lxx_N0.transpose();
    condensed_workspace_[thread_id].q = this->workspace_[N0].lp.tail(nx);
    condensed_workspace_[thread_id].A = this->workspace_[N0].F;
    condensed_workspace_[thread_id].C = this->workspace_[N0].C;
    condensed_workspace_[thread_id].c = this->workspace_[N0].f;
}

// To be deleted
template<class KerD>
void CLQRParallelSolver<KerD>::reduction(int N0, int N1, bool is_last_segment, scalar sigma) {
    
    ParallelCLQRKernel::terminal_step_with_factorization(
        this->model_.knotpoints[N1], this->workspace_[N1], sigma, is_last_segment);
    for (int k = N1 - 1; k >= N0; --k) {
        ParallelCLQRKernel::step_with_factorization(
            this->model_.knotpoints[k], this->workspace_[k], this->workspace_[k + 1], sigma, is_last_segment);
    }

    int nx = this->model_.n;
    ConstMatrixRef Lxx_N0 = this->workspace_[N0].L.bottomRightCorner(nx, nx);   
    condensed_workspace_[0].Q = Lxx_N0 * Lxx_N0.transpose();
    condensed_workspace_[0].q = this->workspace_[N0].lp.tail(nx);
    condensed_workspace_[0].A = this->workspace_[N0].F;
    condensed_workspace_[0].C = this->workspace_[N0].C;
    condensed_workspace_[0].c = this->workspace_[N0].f;
}   

template<class KerD>
void CLQRParallelSolver<KerD>::reduction_without_factorization(int thread_id, scalar sigma) {
    int N0 = idx_start_[thread_id];
    int N1 = N0 + Nseg_[thread_id];
    bool is_last_segment = (thread_id == num_segments_ - 1);

    terminal_workspace_[thread_id] = this->workspace_[N1]; //TODO improve efficiency
    ParallelCLQRKernel::terminal_step_without_factorization(
        this->model_.knotpoints[N1], terminal_workspace_[thread_id], sigma, is_last_segment);
    if (N1 - 1 >= N0) {
        ParallelCLQRKernel::step_without_factorization(
            this->model_.knotpoints[N1 - 1], this->workspace_[N1 - 1], terminal_workspace_[thread_id], sigma, is_last_segment);
        for (int k = N1 - 2; k >= N0; --k) {
            ParallelCLQRKernel::step_without_factorization(
                this->model_.knotpoints[k], this->workspace_[k], this->workspace_[k + 1], sigma, is_last_segment);
        }
    }
    int nx = this->model_.n;
    condensed_workspace_[thread_id].q = this->workspace_[N0].lp.tail(nx);
    condensed_workspace_[thread_id].c = this->workspace_[N0].f;


    // ParallelCLQRKernel::terminal_step_without_factorization(
    //     this->model_.knotpoints[N1], this->workspace_[N1], sigma, is_last_segment);
    // // for (int k = N1 - 1; k >= N0; --k) {
    // //     ParallelCLQRKernel::step_without_factorization(
    // //         this->model_.knotpoints[k], this->workspace_[k], this->workspace_[k + 1], sigma, is_last_segment);
    // // }

    // int nx = this->model_.n;
    // condensed_workspace_[thread_id].q = this->workspace_[N0].lp.tail(nx);
    // condensed_workspace_[thread_id].c = this->workspace_[N0].f;

 
    // ParallelCLQRKernel::terminal_step_with_factorization(
    //     this->model_.knotpoints[N1], this->workspace_[N1], sigma, is_last_segment);
    // for (int k = N1 - 1; k >= N0; --k) {
    //     ParallelCLQRKernel::step_with_factorization(
    //         this->model_.knotpoints[k], this->workspace_[k], this->workspace_[k + 1], sigma, is_last_segment);
    // }
    
    // int nx = this->model_.n;
    // ConstMatrixRef Lxx_N0 = this->workspace_[N0].L.bottomRightCorner(nx, nx);   
    // condensed_workspace_[thread_id].Q = Lxx_N0 * Lxx_N0.transpose();
    // condensed_workspace_[thread_id].q = this->workspace_[N0].lp.tail(nx);
    // condensed_workspace_[thread_id].A = this->workspace_[N0].F;
    // condensed_workspace_[thread_id].C = this->workspace_[N0].C;
    // condensed_workspace_[thread_id].c = this->workspace_[N0].f;

}

template<class KerD>
void CLQRParallelSolver<KerD>::consensus() {
    condensed_workspace_[num_segments_-1].P = condensed_workspace_[num_segments_-1].Q;
    condensed_workspace_[num_segments_-1].p = condensed_workspace_[num_segments_-1].q;

     for (int i = num_segments_ - 2; i >= 0; --i) {
        ConstMatrixRef P_next = condensed_workspace_[i+1].P;
        ConstVectorRef p_next = condensed_workspace_[i+1].p;

        ConstMatrixRef A_i = condensed_workspace_[i].A;
        ConstMatrixRef C_i = condensed_workspace_[i].C;
        ConstMatrixRef Q_i = condensed_workspace_[i].Q;
        ConstVectorRef c_i = condensed_workspace_[i].c;
        ConstVectorRef q_i = condensed_workspace_[i].q;
        MatrixRef P_i      = condensed_workspace_[i].P;
        VectorRef p_i      = condensed_workspace_[i].p;
        MatrixRef PC_i     = condensed_workspace_[i].PC;
        MatrixRef PA_i     = condensed_workspace_[i].PA;
        MatrixRef D_i      = condensed_workspace_[i].D;
        VectorRef Pc       = condensed_workspace_[i].Pc;

        PC_i.noalias() = C_i * P_next;
        PC_i.diagonal().array() += 1.0;
        PA_i.noalias() = P_next * A_i; 

        condensed_workspace_[i].lu_fact.compute(PC_i);
        D_i = condensed_workspace_[i].lu_fact.solve(A_i);
        P_i = Q_i;
        P_i.noalias() += D_i.transpose() * PA_i;

        Pc = p_next;
        Pc.noalias() += P_next * c_i;
        p_i = q_i;
        p_i.noalias() += D_i.transpose() * Pc;
    }
}

template<class KerD>
void CLQRParallelSolver<KerD>::consensus_without_factorization() {
    condensed_workspace_[num_segments_-1].p = condensed_workspace_[num_segments_-1].q;

     for (int i = num_segments_ - 2; i >= 0; --i) {
        ConstMatrixRef P_next = condensed_workspace_[i+1].P;
        ConstVectorRef p_next = condensed_workspace_[i+1].p;

        ConstVectorRef c_i = condensed_workspace_[i].c;
        ConstVectorRef q_i = condensed_workspace_[i].q;
        VectorRef p_i      = condensed_workspace_[i].p;
        MatrixRef D_i      = condensed_workspace_[i].D;
        VectorRef Pc       = condensed_workspace_[i].Pc;

        Pc = p_next;
        Pc.noalias() += P_next * c_i;
        p_i = q_i;
        p_i.noalias() += D_i.transpose() * Pc;
    }
}

template<class KerD>
void CLQRParallelSolver<KerD>::update_dual_variables(scalar alpha) {
    #pragma omp parallel num_threads(num_segments_) 
    {
        int tid = omp_get_thread_num();
        int N0 = idx_start_[tid];
        int N1 = (tid == num_segments_ - 1) ? N0 + Nseg_[tid] : N0 + Nseg_[tid] - 1;
        for (int k = N0; k <= N1; ++k) {    // <= !
            auto& data = this->workspace_[k];
            auto& kpoint = this->model_.knotpoints[k];
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
}

// using ParallelCLQRSolver = CLQRParallelSolver<ParallelCLQRKernelData>;

} // namespace lqr