#pragma once

#include <qdldl/qdldl.h>
#include <Eigen/Sparse>
#include "clqr/lqr_model.hpp"
#include "qdldl_typedefs.hpp"

namespace lqr {

void assign_dense_matrix(Eigen::Index i0, Eigen::Index j0,
                         const MatrixXs& dense_mat,
                         Eigen::SparseMatrix<scalar>& sparse_mat,
                         bool update, bool fill_upper = false) {
    assert(i0 + dense_mat.rows() <= sparse_mat.rows());
    assert(j0 + dense_mat.cols() <= sparse_mat.cols());

    for (Eigen::Index j = 0; j < dense_mat.cols(); ++j) {
        for (Eigen::Index i = 0; i < dense_mat.rows(); ++i) {
            if (!fill_upper || i <= j) {
                if (update) {
                    sparse_mat.coeffRef(i0 + i, j0 + j) = dense_mat(i, j);
                } else {
                    sparse_mat.insert(i0 + i, j0 + j)   = dense_mat(i, j);
                }
            }
        }
    }
}

void assign_diagonal_matrix(Eigen::Index i0, Eigen::Index j0,
                            const VectorXs& diag_vec,
                            Eigen::SparseMatrix<scalar>& sparse_mat,
                            bool update, bool negate = false) {
    scalar sign = negate ? -1.0 : 1.0;
    assert(i0 + diag_vec.size() <= sparse_mat.rows());
    assert(j0 + diag_vec.size() <= sparse_mat.cols());
    for (Eigen::Index i = 0; i < diag_vec.size(); ++i) {
        if (update) {
            sparse_mat.coeffRef(i0 + i, j0 + i) = sign * diag_vec(i);
        } else {
            sparse_mat.insert(i0 + i, j0 + i) = sign * diag_vec(i);
        }
    }    
}

void assign_diagonal_matrix(Eigen::Index i0, Eigen::Index j0,
                            scalar value, scalar size,
                            Eigen::SparseMatrix<scalar>& sparse_mat,
                            bool update) {
    assert(i0 + size <= sparse_mat.rows());
    assert(j0 + size <= sparse_mat.cols());
    for (Eigen::Index i = 0; i < size; ++i) {
        if (update) {
            sparse_mat.coeffRef(i0 + i, j0 + i) = value;
        } else {
            sparse_mat.insert(i0 + i, j0 + i)   = value;
        }
    }
}

class KKTSystem {
public:
    KKTSystem(int nx, int nu, int N, const std::vector<int>& ncs) {
        int nxu = nx + nu;
        int num_rows = 0;
        num_rows += nxu + ncs[0]; // stage 0
        for (int k = 1; k < N; ++k) {
            num_rows += nxu + ncs[k] + nx; // stage k
        }
        num_rows += nx + ncs[N]; // terminal stage
        KKT_mat_.resize(num_rows, num_rows);
        rhs_.resize(num_rows);

        Hs_tmp_.resize(N + 1);
        for (int k = 0; k < N; ++k) { Hs_tmp_[k].resize(nxu, nxu); }
        Hs_tmp_[N].resize(nx, nx);
    }
    
    void fill_stage_cost_matrices(const MatrixXs& H, 
                                  int nx, int nu, 
                                  int row_offset, bool update) {
        ConstMatrixRef Qk  = H.bottomRightCorner(nx, nx);
        ConstMatrixRef Rk  = H.topLeftCorner(nu, nu);
        ConstMatrixRef Stk = H.bottomLeftCorner(nx, nu);
        assign_dense_matrix(row_offset, row_offset, Qk, KKT_mat_, update, true);
        assign_dense_matrix(row_offset, row_offset + nx, Stk, KKT_mat_, update);
        assign_dense_matrix(row_offset + nx, row_offset + nx, Rk, KKT_mat_, update, true);
    }

    void fill_stage_dynamics_matrices(const MatrixXs& E,
                                      int nx, int nu, int nc, 
                                      int row_offset, scalar rho_dyn, bool update) {
        const int nxu  = nx + nu;
        ConstMatrixRef Ak = E.rightCols(nx);
        ConstMatrixRef Bk = E.leftCols(nu);
        // Matrix A
        assign_dense_matrix(row_offset, row_offset + nxu, Ak.transpose(), KKT_mat_, update);
        // Matrix B
        assign_dense_matrix(row_offset + nx, row_offset + nxu, Bk.transpose(), KKT_mat_, update);
        // -rho_dyn * I for dynamics regularization
        assign_diagonal_matrix(row_offset + nxu, row_offset + nxu, -rho_dyn, nx, KKT_mat_, update);
        // -I coupling to next stage
        assign_diagonal_matrix(row_offset + nxu, row_offset + nxu + nx + nc, -1.0, nx, KKT_mat_, update);
    }

    void fill_stage_constraint_matrices(const MatrixXs& D_con,
                                        int nx, int nu, int nc,
                                        int row_offset, bool update) {
        if (nc > 0) {
            const int nxu = nx + nu;
            ConstMatrixRef Dxk = D_con.rightCols(nx);
            ConstMatrixRef Duk = D_con.leftCols(nu);
            // Dxk
            assign_dense_matrix(row_offset, row_offset + nxu + nx, Dxk.transpose(), KKT_mat_, update);
            // Duk
            assign_dense_matrix(row_offset + nx, row_offset + nxu + nx, Duk.transpose(), KKT_mat_, update);
            // -rho_ineq * I for inequality regularization
            assign_diagonal_matrix(row_offset + nxu + nx, row_offset + nxu + nx, -1.0, nc, KKT_mat_, update);
        }
    }

    void update_rho_vecs(const LQRModel& model,
                         const std::vector<VectorXs>& inv_rho_vecs) 
    {
        int nx  = model.n;
        int nu  = model.m;
        int nxu = nx + nu;
        int N   = model.N;
        int row_offset = 0;
        // Stage 0
        {
            const auto& kpoint = model.get_node(0);
            int nc = kpoint.get_constraint_dim();
            if (nc > 0) {
                assign_diagonal_matrix(row_offset + nxu, row_offset + nxu,
                                       inv_rho_vecs[0], KKT_mat_, true, true);
            }
            row_offset += nxu + nc;
        }
        // Intermediate stages k = 1, ..., N-1
        for (int k = 1; k < N; ++k) {
            const auto& kpoint = model.get_node(k); 
            int nc = kpoint.get_constraint_dim();
            if (nc > 0) {
                assign_diagonal_matrix(row_offset + nxu + nx, row_offset + nxu + nx,
                                       inv_rho_vecs[k], KKT_mat_, true, true);
            }
            row_offset += nxu + nx + nc;                
        }
        // Terminal stage
        {
            const auto& terminal_kpoint = model.get_node(N);
            int nc = terminal_kpoint.get_constraint_dim();   
            if (nc > 0) {
                assign_diagonal_matrix(row_offset + nx, row_offset + nx,
                                       inv_rho_vecs[N], KKT_mat_, true, true);
            }
        }        
    }

    void form_KKT_system(const LQRModel& model, scalar rho_dyn, scalar sigma, bool update) 
    {
        if (!update) {
            KKT_mat_.setZero();
        }
        rhs_.setZero();
        int nx  = model.n;
        int nu  = model.m;
        int nxu = nx + nu;
        int N   = model.N;
        int row_offset = 0;
        /* 
        ** Note: just filling the upper triangular part 
        */
        /*
        ** Stage 0: [R0  B0'         Du0'          0  ]
        **          [B0 -rho_dyn * I  0            -I  ]
        **          [Du0 0          -rho_ineq * I  0  ]
        */
        {
            const auto& kpoint = model.get_node(0);
            const int nc = kpoint.get_constraint_dim();
            Hs_tmp_[0] = kpoint.H;
            Hs_tmp_[0].diagonal().array() += sigma;
            // Cost matrix R0 for stage 0
            ConstMatrixRef R0 = Hs_tmp_[0].topLeftCorner(nu, nu);
            assign_dense_matrix(row_offset, row_offset, R0, KKT_mat_, update, true);
            // Matrix B0 
            ConstMatrixRef B0 = kpoint.E.leftCols(nu);
            assign_dense_matrix(row_offset, row_offset + nu, B0.transpose(), KKT_mat_, update);
            // -rho_dyn * I for dynamics regularization
            assign_diagonal_matrix(row_offset + nu, row_offset + nu, -rho_dyn, nx, KKT_mat_, update);
            // -I coupling to next stage
            assign_diagonal_matrix(row_offset + nu, row_offset + nxu + nc, -1.0, nx, KKT_mat_, update);
            // Inequality constraints: Du0 and -rho_ineq * I
            if (nc > 0) {
                ConstMatrixRef Du0 = kpoint.D_con.leftCols(nu);
                assign_dense_matrix(row_offset, row_offset + nxu, Du0.transpose(), KKT_mat_, update);
                // -rho_ineq * I for inequality regularization
                assign_diagonal_matrix(row_offset + nxu, row_offset + nxu, -1.0, nc, KKT_mat_, update);
            } 
            row_offset += nxu + nc;
        }
        // Intermediate stages k = 1, ..., N-1
        for (int k = 1; k < N; ++k) {
            const auto& kpoint = model.get_node(k); 
            const int nc = kpoint.get_constraint_dim();
            /*
            ** Stage k: [-I 0 Qk  Sk'  Ak'         Dxk'          0]
            **          [ 0 0 Sk  Rk   Bk'         Duk'          0]
            **          [ 0 0 Ak  Bk  -rho_dyn * I  0            -I]
            **          [ 0 0 Dxk Duk  0          -rho_ineq * I  0]
            */
            // Cost matrix H = [R S; S' Q] 
            Hs_tmp_[k] = kpoint.H;
            Hs_tmp_[k].diagonal().array() += sigma;
            fill_stage_cost_matrices(Hs_tmp_[k], nx, nu, row_offset, update);
            // Dynamics matrices E = [B A]
            fill_stage_dynamics_matrices(kpoint.E, nx, nu, nc, row_offset, rho_dyn, update);
            // Constraint matrices D_con = [Du Dx]
            fill_stage_constraint_matrices(kpoint.D_con, nx, nu, nc, row_offset, update);
            
            row_offset += nxu + nx + nc;                
        }
        /* Terminal stage: [-I 0 QN   DxN'          ]
        **                 [ 0 0 DxN -rho_ineq * I ]
        */
        {
            const auto& terminal_kpoint = model.get_node(N);
            const int nc = terminal_kpoint.get_constraint_dim();   
            // Terminal cost matrix QN
            Hs_tmp_[N] = terminal_kpoint.H;
            Hs_tmp_[N].diagonal().array() += sigma;
            assign_dense_matrix(row_offset, row_offset, Hs_tmp_[N], KKT_mat_, update, true);
            // Terminal constraints
            if (nc > 0) {
                ConstMatrixRef DxN = terminal_kpoint.D_con;
                assign_dense_matrix(row_offset, row_offset + nx, DxN.transpose(), KKT_mat_, update);
                // -rho_ineq * I for terminal inequality regularization
                assign_diagonal_matrix(row_offset + nx, row_offset + nx, -1.0, nc, KKT_mat_, update);
            }
        }
    }

    void update_initial_stage_rhs(const LQRModel& model, const VectorXs& x0) {
        const auto& kpoint = model.get_node(0);
        const int nx  = model.n;
        const int nu  = model.m;
        const int nc = kpoint.get_constraint_dim();
        ConstMatrixRef S0 = kpoint.H.topRightCorner(nu, nx);
        ConstMatrixRef A0 = kpoint.E.rightCols(nx);
        rhs_.head(nu).noalias() += -S0 * x0;
        rhs_.segment(nu, nx).noalias() += -A0 * x0;
        // if (nc > 0) {
        //     ConstMatrixRef Dx0 = kpoint.D_con.rightCols(nx);
        //     rhs_.segment(nxu, nc).noalias() -= Dx0 * x0;
        // }            
    }
     
    void form_rhs(const LQRModel& model,
                  const std::vector<VectorXs>& ws,
                  const std::vector<VectorXs>& ys,
                  const std::vector<VectorXs>& zs,
                  const std::vector<VectorXs>& inv_rho_vecs, const scalar sigma) 
    {
        int nx  = model.n;
        int nu  = model.m;
        int nxu = nx + nu;
        int N   = model.N;
        int row_offset = 0;
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

            // rhs_.segment(nu, nx).noalias() = -A0 * x0;
            rhs_.segment(nu, nx) = -c0;
            
            if (nc > 0) {
                // ConstMatrixRef Dx0 = kpoint.D_con.rightCols(nx);
                rhs_.segment(nxu, nc) = zs[0];
                rhs_.segment(nxu, nc).noalias() -= inv_rho_vecs[0].cwiseProduct(ys[0]);
                // rhs_.segment(nxu, nc).noalias() -= Dx0 * x0;
            }            
            row_offset += nxu + nc;
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
            rhs_.segment(row_offset, nx).noalias()  = -qk;
            rhs_.segment(row_offset, nx).noalias() += sigma * xs;
            
            rhs_.segment(row_offset + nx, nu).noalias()  = -rk;
            rhs_.segment(row_offset + nx, nu).noalias() += sigma * us;
            
            rhs_.segment(row_offset + nxu, nx).noalias() = -ck;

            if (nc > 0) {
                rhs_.segment(row_offset + nxu + nx, nc) = zs[k];
                rhs_.segment(row_offset + nxu + nx, nc).noalias() -= inv_rho_vecs[k].cwiseProduct(ys[k]);
            }
            row_offset += nxu + nx + nc;
        }
        // Terminal stage
        {
            const auto& terminal_kpoint = model.get_node(N);
            const int nc = terminal_kpoint.get_constraint_dim();   
            ConstVectorRef qN = terminal_kpoint.h;
            ConstVectorRef xN = ws[N];
            rhs_.segment(row_offset, nx).noalias()  = -qN;
            rhs_.segment(row_offset, nx).noalias() += sigma * xN;   
            if (nc > 0) {
                rhs_.segment(row_offset + nx, nc) = zs[N];
                rhs_.segment(row_offset + nx, nc).noalias() -= inv_rho_vecs[N].cwiseProduct(ys[N]);
            }
        }
    }

    std::unique_ptr<CscMatrix> get_KKT_csc_matrix() {
        // Ensure type compatibility at compile time
        // static_assert(sizeof(QDLDL_int) == sizeof(Eigen::StorageIndex), 
        //     "QDLDL_int size must match Eigen::StorageIndex size");
        // static_assert(sizeof(QDLDL_float) == sizeof(scalar), 
        //     "QDLDL_float size must match scalar size");

        if (!KKT_mat_.isCompressed()) { KKT_mat_.makeCompressed(); }
        
        // Extract upper triangular part properly
        // Note: triangularView returns a view, need to convert to concrete matrix
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

    const VectorXs& get_rhs() const { return rhs_; }
    
private:
    // std::vector<Eigen::Triplet<scalar>> triplets_;
    // int max_num_triplets_;
    Eigen::SparseMatrix<scalar> KKT_mat_;
    // Eigen::SparseMatrix<scalar> KKT_mat_triu_; 
    VectorXs rhs_;
    
    std::vector<MatrixXs> Hs_tmp_;
}; 

} // namespace lqr