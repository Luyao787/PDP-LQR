#pragma once

#include <Eigen/OrderingMethods>
#include "utils.hpp"

namespace lqr
{
class KKTSystem {
public:
    KKTSystem(int nx, int nu, int N, const std::vector<int>& ncs);

    void fill_stage_cost_matrices(const MatrixXs& H, 
                                  int nx, int nu, 
                                  int row_offset, 
                                  bool update, bool ignore_zeros=true);
    void fill_stage_dynamics_matrices(const MatrixXs& E,
                                      int nx, int nu, int nc, 
                                      int row_offset, int col_offset,
                                      bool update);
    void fill_stage_constraint_matrices(const MatrixXs& D_con,
                                        int nx, int nu, int nc,
                                        int row_offset, int col_offset,
                                        bool update, bool ignore_zeros=true);
    void update_rho_vecs(const LQRModel& model, const std::vector<VectorXs>& inv_rho_vecs);
    void form_KKT_matrix(const LQRModel& model, scalar rho_dyn, scalar sigma, bool update);
    void update_rhs_initial_stage(const LQRModel& model, const VectorXs& x0);
    void form_rhs(const LQRModel& model,
                  const std::vector<VectorXs>& ws,
                  const std::vector<VectorXs>& ys,
                  const std::vector<VectorXs>& zs,
                  const std::vector<VectorXs>& inv_rho_vecs, const scalar sigma);
    std::unique_ptr<CscMatrix> get_KKT_csc_matrix();
    const VectorXs& get_rhs() const { return rhs_; }

private:
    Eigen::SparseMatrix<scalar> KKT_mat_;
    VectorXs rhs_;
    std::vector<int> ncs_;    
    std::vector<MatrixXs> Hs_tmp_;
};  

KKTSystem::KKTSystem(int nx, int nu, int N, const std::vector<int>& ncs)
    : ncs_(N + 1) {
    int nxu = nx + nu;
    int num_rows = 0;
    num_rows += nxu + ncs[0]; // stage 0
    ncs_[0] = ncs[0];
    for (int k = 1; k < N; ++k) {
        num_rows += nxu + ncs[k] + nx; // stage k
        ncs_[k] = ncs[k];
    }
    num_rows += nx + ncs[N]; // terminal stage
    ncs_[N] = ncs[N];
    KKT_mat_.resize(num_rows, num_rows);
    rhs_.resize(num_rows);

    Hs_tmp_.resize(N + 1);
    for (int k = 0; k < N; ++k) { Hs_tmp_[k].resize(nxu, nxu); }
    Hs_tmp_[N].resize(nx, nx);
}

void KKTSystem::fill_stage_cost_matrices(const MatrixXs& H, 
                                         int nx, int nu, 
                                         int row_offset, 
                                         bool update, bool ignore_zeros) {
    ConstMatrixRef Qk  = H.bottomRightCorner(nx, nx);
    ConstMatrixRef Rk  = H.topLeftCorner(nu, nu);
    ConstMatrixRef Stk = H.bottomLeftCorner(nx, nu);
    assign_dense_matrix(row_offset, row_offset, Qk, KKT_mat_, update, true, ignore_zeros);
    assign_dense_matrix(row_offset, row_offset + nx, Stk, KKT_mat_, update, false, ignore_zeros);
    assign_dense_matrix(row_offset + nx, row_offset + nx, Rk, KKT_mat_, update, true, ignore_zeros);
}

void KKTSystem::fill_stage_dynamics_matrices(const MatrixXs& E,
                                             int nx, int nu, int nc, 
                                             int row_offset, int col_offset,
                                             bool update) {
    ConstMatrixRef Ak = E.rightCols(nx);
    ConstMatrixRef Bk = E.leftCols(nu);
    // -I
    assign_diagonal_matrix(row_offset, col_offset, -1.0, nx, KKT_mat_, update);
    // Matrix A
    assign_dense_matrix(row_offset, col_offset + nx + nc, Ak.transpose(), KKT_mat_, update);
    // Matrix B
    assign_dense_matrix(row_offset + nx, col_offset + nx + nc, Bk.transpose(), KKT_mat_, update);
}

void KKTSystem::fill_stage_constraint_matrices(const MatrixXs& D_con,
                                               int nx, int nu, int nc,
                                               int row_offset, int col_offset,
                                               bool update, bool ignore_zeros) {
    if (nc > 0) {
        ConstMatrixRef Dxk = D_con.rightCols(nx);
        ConstMatrixRef Duk = D_con.leftCols(nu);
        // Dxk
        assign_dense_matrix(row_offset, col_offset + nx, Dxk.transpose(), KKT_mat_, update, false, ignore_zeros);
        // Duk
        assign_dense_matrix(row_offset + nx, col_offset + nx, Duk.transpose(), KKT_mat_, update, false, ignore_zeros);
    }
}

void KKTSystem::update_rho_vecs(const LQRModel& model,
                                const std::vector<VectorXs>& inv_rho_vecs) 
{
    int nx  = model.n;
    int nu  = model.m;
    int nxu = nx + nu;
    int N   = model.N;
    int row_offset = N * nxu;
    for (int k = 0; k < N + 1; ++k) {
        const auto& kpoint = model.get_node(k); 
        int nc = kpoint.get_constraint_dim();
        if (nc > 0) {
            assign_diagonal_matrix(row_offset, row_offset,
                                   inv_rho_vecs[k], KKT_mat_, true, true);
        }
        row_offset += nc + nx;       
    }
}

void KKTSystem::form_KKT_matrix(const LQRModel& model, scalar rho_dyn, scalar sigma, bool update) 
{
    if (!update) {
        KKT_mat_.setZero();
    }
    int nx  = model.n;
    int nu  = model.m;
    int nxu = nx + nu;
    int N   = model.N;
    int row_offset = 0;
    int col_offset = N * nxu; 
    /* 
    ** Note: just filling the upper triangular part 
    */
    {
        const auto& kpoint = model.get_node(0);
        const int nc = kpoint.get_constraint_dim();
        Hs_tmp_[0] = kpoint.H;
        Hs_tmp_[0].diagonal().array() += sigma;
        // Cost matrix R0 for stage 0
        ConstMatrixRef R0 = Hs_tmp_[0].topLeftCorner(nu, nu);
        assign_dense_matrix(row_offset, row_offset, R0, KKT_mat_, update, true);

        // D_{u,0}^T
        if (nc > 0) {
            ConstMatrixRef Du0 = kpoint.D_con.leftCols(nu);
            assign_dense_matrix(row_offset, col_offset, Du0.transpose(), KKT_mat_, update);
        }

        // B0
        ConstMatrixRef B0 = kpoint.E.leftCols(nu);  
        assign_dense_matrix(row_offset, col_offset + nc, B0.transpose(), KKT_mat_, update);

        row_offset += nu;
        col_offset += nc;
    }
    // Intermediate stages k = 1, ..., N-1
    for (int k = 1; k < N; ++k) {
        const auto& kpoint = model.get_node(k); 
        const int nc = kpoint.get_constraint_dim();
        // Cost matrix H = [R S; S' Q] 
        Hs_tmp_[k] = kpoint.H;
        Hs_tmp_[k].diagonal().array() += sigma;

        fill_stage_cost_matrices(Hs_tmp_[k], nx, nu, row_offset, update);

        fill_stage_dynamics_matrices(kpoint.E, nx, nu, nc, row_offset, col_offset, update);

        fill_stage_constraint_matrices(kpoint.D_con, nx, nu, nc, row_offset, col_offset, update);
        
        row_offset += nxu;
        col_offset += nc + nx;
    }
    /* Terminal stage: */
    {
        const auto& terminal_kpoint = model.get_node(N);
        const int nc = terminal_kpoint.get_constraint_dim();   
        // Terminal cost matrix QN
        Hs_tmp_[N] = terminal_kpoint.H;
        Hs_tmp_[N].diagonal().array() += sigma;
        assign_dense_matrix(row_offset, row_offset, Hs_tmp_[N], KKT_mat_, update, true);
        // -I
        assign_diagonal_matrix(row_offset, col_offset, -1.0, nx, KKT_mat_, update);
        // Terminal constraints
        if (nc > 0) {
            ConstMatrixRef DxN = terminal_kpoint.D_con;
            assign_dense_matrix(row_offset, col_offset + nx, DxN.transpose(), KKT_mat_, update);
        }
        row_offset += nx;
    }
    // Regularization
    {   
        assign_diagonal_matrix(row_offset, row_offset, -1.0, ncs_[0], KKT_mat_, update);
        row_offset += ncs_[0];
        for (int k = 1; k < N + 1; ++k) {
            assign_diagonal_matrix(row_offset, row_offset, -rho_dyn, nx, KKT_mat_, update);
            row_offset += nx;  
            assign_diagonal_matrix(row_offset, row_offset, -1.0, ncs_[k], KKT_mat_, update);
            row_offset += ncs_[k];
        }
    }
}

void KKTSystem::update_rhs_initial_stage(const LQRModel& model, const VectorXs& x0) {
    const auto& kpoint = model.get_node(0);
    const int N  = model.N;
    const int nx = model.n;
    const int nu = model.m;
    const int nc = kpoint.get_constraint_dim();
    ConstMatrixRef S0 = kpoint.H.topRightCorner(nu, nx);
    ConstMatrixRef A0 = kpoint.E.rightCols(nx);
    rhs_.head(nu).noalias() += -S0 * x0;
    int row_offset = N * (nx + nu) + nc;
    rhs_.segment(row_offset, nx).noalias() += -A0 * x0;
    // if (nc > 0) {
    //     ConstMatrixRef Dx0 = kpoint.D_con.rightCols(nx);
    //     rhs_.segment(nxu, nc).noalias() -= Dx0 * x0;
    // }            
}
    
void KKTSystem::form_rhs(const LQRModel& model,
                         const std::vector<VectorXs>& ws,
                         const std::vector<VectorXs>& ys,
                         const std::vector<VectorXs>& zs,
                         const std::vector<VectorXs>& inv_rho_vecs, scalar sigma) 
{
    int nx  = model.n;
    int nu  = model.m;
    int nxu = nx + nu;
    int N   = model.N;
    int row_offset1 = 0;
    int row_offset2 = N * nxu;
    // Stage 0
    {
        const auto& kpoint = model.get_node(0);
        const int nc = kpoint.get_constraint_dim();

        // ConstMatrixRef S0 = kpoint.H.topRightCorner(nu, nx);
        // ConstMatrixRef A0 = kpoint.E.rightCols(nx);
        ConstVectorRef r0 = kpoint.h.head(nu);
        ConstVectorRef c0 = kpoint.c;
        // rhs_.head(nu).noalias() = -S0 * x0;
        rhs_.head(nu) = -r0;
        rhs_.head(nu).noalias() += sigma * ws[0].head(nu);

        if (nc > 0) {
            // ConstMatrixRef Dx0 = kpoint.D_con.rightCols(nx);
            rhs_.segment(row_offset2, nc) = zs[0];
            rhs_.segment(row_offset2, nc).noalias() -= inv_rho_vecs[0].cwiseProduct(ys[0]);
            // rhs_.segment(nxu, nc).noalias() -= Dx0 * x0;
        }            
        // rhs_.segment(nu, nx).noalias() = -A0 * x0;
        rhs_.segment(row_offset2 + nc, nx) = -c0;
                        
        row_offset1 += nu;
        row_offset2 += nc + nx;
    }
    // Intermediate stages k = 1, ..., N-1
    for (int k = 1; k < N; ++k) {
        const auto& kpoint = model.get_node(k); 
        const int nc = kpoint.get_constraint_dim();
        ConstVectorRef qk = kpoint.h.tail(nx);
        ConstVectorRef rk = kpoint.h.head(nu);
        ConstVectorRef ck = kpoint.c;
        ConstVectorRef xs = ws[k].tail(nx);
        ConstVectorRef us = ws[k].head(nu);

        rhs_.segment(row_offset1, nx).noalias()  = -qk;
        rhs_.segment(row_offset1, nx).noalias() += sigma * xs;
        
        rhs_.segment(row_offset1 + nx, nu).noalias()  = -rk;
        rhs_.segment(row_offset1 + nx, nu).noalias() += sigma * us;
        
        if (nc > 0) {
            rhs_.segment(row_offset2, nc) = zs[k];
            rhs_.segment(row_offset2, nc).noalias() -= inv_rho_vecs[k].cwiseProduct(ys[k]);
        }
        
        rhs_.segment(row_offset2 + nc, nx).noalias() = -ck;
        
        row_offset1 += nxu;
        row_offset2 += nc + nx;
    }
    // Terminal stage
    {
        const auto& terminal_kpoint = model.get_node(N);
        const int nc = terminal_kpoint.get_constraint_dim();   
        ConstVectorRef qN = terminal_kpoint.h;
        ConstVectorRef xN = ws[N];
        rhs_.segment(row_offset1, nx).noalias()  = -qN;
        rhs_.segment(row_offset1, nx).noalias() += sigma * xN;   
        if (nc > 0) {
            rhs_.segment(row_offset2, nc) = zs[N];
            rhs_.segment(row_offset2, nc).noalias() -= inv_rho_vecs[N].cwiseProduct(ys[N]);
        }
    }
}

std::unique_ptr<CscMatrix> KKTSystem::get_KKT_csc_matrix() {
    // Ensure type compatibility at compile time
    // static_assert(sizeof(QDLDL_int) == sizeof(Eigen::StorageIndex), 
    //     "QDLDL_int size must match Eigen::StorageIndex size");
    // static_assert(sizeof(QDLDL_float) == sizeof(scalar), 
    //     "QDLDL_float size must match scalar size");

    if (!KKT_mat_.isCompressed()) { KKT_mat_.makeCompressed(); }
    
    // KKT_mat_triu_ = KKT_mat_.triangularView<Eigen::Upper>();

    auto K_csc = std::make_unique<CscMatrix>();
    
    K_csc->m     = static_cast<QDLDL_int>(KKT_mat_.rows());
    K_csc->n     = static_cast<QDLDL_int>(KKT_mat_.cols());
    K_csc->nzmax = static_cast<QDLDL_int>(KKT_mat_.nonZeros());
    
    // Safe type-checked cast to QDLDL types
    K_csc->p = reinterpret_cast<const QDLDL_int*>(KKT_mat_.outerIndexPtr());
    K_csc->i = reinterpret_cast<const QDLDL_int*>(KKT_mat_.innerIndexPtr());
    K_csc->x = reinterpret_cast<const QDLDL_float*>(KKT_mat_.valuePtr());
    
    return K_csc;
}

} // namespace lqr