#pragma once

#include <vector>
#include "clqr/typedefs.hpp"

namespace lqr {

class CondensedSystemSolverBase {
public:
    virtual ~CondensedSystemSolverBase() = default;
    
    virtual bool backward() = 0;
    
    virtual void forward(const VectorXs& ) = 0;

};

class CondensedSystemLUSolver : public CondensedSystemSolverBase {
private:
    struct KernelData {
        MatrixXs A, C, P;
        VectorXs c, p;

        MatrixXs PC, PA, D;
        Eigen::PartialPivLU<MatrixXs> lu_fact;
        VectorXs c_bar;

        VectorXs xhat, uhat;

        KernelData(int nx)
            : A(nx, nx), C(nx, nx), P(nx, nx),
              c(nx), p(nx),
              PC(nx, nx), PA(nx, nx), D(nx, nx),
              c_bar(nx),
              xhat(nx), uhat(nx) {}
    };

    int num_segments_;
    std::vector<KernelData> workspace_;

public:
    CondensedSystemLUSolver(int nx, int num_segments) 
        : num_segments_(num_segments) {
        workspace_.reserve(num_segments);
        for (int i = 0; i < num_segments; ++i) {
            workspace_.emplace_back(nx);
        }
    }

    void update_segment_data(const MatrixXs& Lxx,
                             const MatrixXs& A,
                             const MatrixXs& C,
                             const VectorXs& p,
                             const VectorXs& c, int segment_id) {
        workspace_[segment_id].P.noalias() = Lxx * Lxx.transpose();
        workspace_[segment_id].A = A;
        workspace_[segment_id].C = C;
        workspace_[segment_id].p = p;
        workspace_[segment_id].c = c;
    }

    void update_segment_data(const VectorXs& p,
                             const VectorXs& c, int segment_id) {
        workspace_[segment_id].p = p;
        workspace_[segment_id].c = c;
    }
        
    bool backward() override {
        for (int i = num_segments_ - 2; i >= 0; --i) {
            const auto& P_next = workspace_[i+1].P;
            const auto& A_i    = workspace_[i].A;
            const auto& C_i    = workspace_[i].C;
            
            auto& P_i  = workspace_[i].P;
            auto& PC_i = workspace_[i].PC;
            auto& PA_i = workspace_[i].PA;
            auto& D_i  = workspace_[i].D;

            PC_i.noalias() = C_i * P_next;
            PC_i.diagonal().array() += 1.0;
            PA_i.noalias() = P_next * A_i; 

            workspace_[i].lu_fact.compute(PC_i);
            D_i = workspace_[i].lu_fact.solve(A_i);
            P_i.noalias() += D_i.transpose() * PA_i;
        }

        return true;
    }

    void forward(const VectorXs& x0) override {
        for (int i = num_segments_ - 2; i >= 0; --i) {
            const auto& P_next = workspace_[i+1].P;
            const auto& p_next = workspace_[i+1].p;
            const auto& D_i = workspace_[i].D;
            const auto& c_i = workspace_[i].c;

            auto& p_i = workspace_[i].p;
            auto& c_bar_i = workspace_[i].c_bar;

            c_bar_i = p_next;
            c_bar_i.noalias() += P_next * c_i;
            p_i.noalias() += D_i.transpose() * c_bar_i;  
        }
        
        workspace_[0].xhat = x0;
        for (int i = 0; i < num_segments_ - 1; ++i) { 
            const auto& P_next = workspace_[i+1].P;
            const auto& p_next = workspace_[i+1].p;
            const auto& A_i = workspace_[i].A;
            const auto& C_i = workspace_[i].C;
            
            auto& c_i = workspace_[i].c;
            auto& xhat  = workspace_[i].xhat;
            auto& uhat  = workspace_[i].uhat;
            auto& xhat_next = workspace_[i+1].xhat;

            c_i.noalias() += A_i * xhat;
            c_i.noalias() -= C_i * p_next;
            xhat_next = workspace_[i].lu_fact.solve(c_i);
            uhat = p_next;
            uhat.noalias() += P_next * xhat_next;
        }  
    }

    const VectorXs& get_xhat(int segment_id) {
        return workspace_[segment_id].xhat;
    }

    const VectorXs& get_uhat(int segment_id) {
        return workspace_[segment_id].uhat;
    }
};
/*
**********************
*/
class CondensedSystemCholeskySolver : public CondensedSystemSolverBase {
private:
    struct KernelData {
        MatrixXs A, C, P, At, Pinv;
        VectorXs c, p;
        VectorXs xhat, uhat;
        Eigen::LLT<MatrixXs> P_chol_fact;
        Eigen::LLT<MatrixXs> C_chol_fact;

        KernelData(int nx)
            : A(nx, nx), C(nx, nx), P(nx, nx), At(nx, nx), Pinv(nx, nx),
              c(nx), p(nx),
              xhat(nx), uhat(nx) {
            // Pre-allocate LLT factorizations with dummy matrices to ensure memory allocation
            Pinv.setIdentity();
            P_chol_fact.compute(Pinv);
            C_chol_fact.compute(Pinv);
        }  
    };

    int num_segments_;
    std::vector<KernelData> workspace_;

public:
    CondensedSystemCholeskySolver(int nx, int num_segments) 
        : num_segments_(num_segments) {
        workspace_.reserve(num_segments);
        for (int i = 0; i < num_segments; ++i) {
            workspace_.emplace_back(nx);
        }
    }

    void update_segment_data(const MatrixXs& Lxx,
                             const MatrixXs& A,
                             const MatrixXs& C,
                             const VectorXs& p,
                             const VectorXs& c, int segment_id) {
        workspace_[segment_id].P.noalias() = Lxx * Lxx.transpose();
        workspace_[segment_id].A = A;
        workspace_[segment_id].C = C;
        workspace_[segment_id].p = p;
        workspace_[segment_id].c = c;
        workspace_[segment_id].At = A.transpose();
        workspace_[segment_id].Pinv.setIdentity();
    }

    void update_segment_data(const VectorXs& p,
                             const VectorXs& c, int segment_id) {
        workspace_[segment_id].p = p;
        workspace_[segment_id].c = c;
    }
        
    bool backward() override {
        for (int i = num_segments_ - 2; i >= 1; --i) {
            const auto& P_next = workspace_[i+1].P;
            const auto& At_i = workspace_[i].At;

            auto& P_i = workspace_[i].P;
            auto& C_i = workspace_[i].C;
            auto& A_i = workspace_[i].A;
            auto& Pinv_i = workspace_[i+1].Pinv;
            auto& P_chol_fact = workspace_[i+1].P_chol_fact;
            auto& C_chol_fact = workspace_[i].C_chol_fact;

            // compute P_next^{-1}
            P_chol_fact.compute(P_next);
            if (P_chol_fact.info() != Eigen::Success) {
                return false;
            }
            P_chol_fact.solveInPlace(Pinv_i);
            //
            C_i.noalias() += Pinv_i;
            C_chol_fact.compute(C_i);
            if (C_chol_fact.info() != Eigen::Success) {
                return false;
            }
            C_chol_fact.solveInPlace(A_i);
            P_i.noalias() += At_i * A_i;
        }
        {
            const auto& P_next = workspace_[1].P;
            auto& C_i = workspace_[0].C;
            auto& Pinv_i = workspace_[1].Pinv;
            auto& P_chol_fact = workspace_[1].P_chol_fact;
            auto& C_chol_fact = workspace_[0].C_chol_fact;
            
            P_chol_fact.compute(P_next);
            if (P_chol_fact.info() != Eigen::Success) {
                return false;
            }
            P_chol_fact.solveInPlace(Pinv_i);
            //
            C_i.noalias() += Pinv_i;
            C_chol_fact.compute(C_i);
            if (C_chol_fact.info() != Eigen::Success) {
                return false;
            }
        }
        return true;
    }

    void forward(const VectorXs& x0) override {
        for (int i = num_segments_ - 2; i >= 1; --i) {
            const auto& A_i = workspace_[i].A;
            auto& p_next = workspace_[i+1].p;
            auto& c_i = workspace_[i].c;
            auto& p_i = workspace_[i].p;
            auto& P_chol_fact = workspace_[i+1].P_chol_fact;

            P_chol_fact.solveInPlace(p_next);
            c_i += p_next;
            p_i.noalias() += A_i.transpose() * c_i;
        }
        {
            auto& c_i = workspace_[0].c;
            auto& p_next = workspace_[1].p;
            auto& P_chol_fact = workspace_[1].P_chol_fact;
            
            P_chol_fact.solveInPlace(p_next);
            c_i += p_next;
        }
        workspace_[0].xhat = x0;
        for (int i = 0; i < num_segments_ - 1; ++i) {
            const auto& Pinv_next = workspace_[i+1].Pinv;
            const auto& p_next = workspace_[i+1].p;

            const auto& c_i = workspace_[i].c;
            const auto& At_i = workspace_[i].At;
            auto& xhat  = workspace_[i].xhat;
            auto& uhat  = workspace_[i].uhat;
            auto& xhat_next = workspace_[i+1].xhat;
            auto& C_chol_fact = workspace_[i].C_chol_fact;
            
            uhat = c_i;
            uhat.noalias() += At_i.transpose() * xhat;
            C_chol_fact.solveInPlace(uhat);
            xhat_next = -p_next;
            xhat_next.noalias() += Pinv_next * uhat;
        }
    }

    const VectorXs& get_xhat(int segment_id) {
        return workspace_[segment_id].xhat;
    }

    const VectorXs& get_uhat(int segment_id) {
        return workspace_[segment_id].uhat;
    }
};

} // namespace lqr