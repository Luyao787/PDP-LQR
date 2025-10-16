#pragma once

#include <qdldl.h>
#include <Eigen/Sparse>
#include "lqr/lqr_model.hpp"


namespace lqr {

// struct CscMatrix {
//     QDLDL_int    m;     // number of rows
//     QDLDL_int    n;     // number of columns
//     QDLDL_int*   p;     // column pointers
//     QDLDL_int*   i;     // row indices
//     QDLDL_float* x;     // non-zero values
//     QDLDL_int    nzmax; // maximum number of non-zero entries
// };

void assign_dense_matrix(Eigen::Index i0, Eigen::Index j0,
                         const MatrixXs& dense_mat,
                         Eigen::SparseMatrix<scalar>& sparse_mat,
                         bool update) {
    assert(i0 + dense_mat.rows() <= sparse_mat.rows());
    assert(j0 + dense_mat.cols() <= sparse_mat.cols());
    for (Eigen::Index j = 0; j < dense_mat.cols(); ++j) {
        for (Eigen::Index i = 0; i < dense_mat.rows(); ++i) {
            if (update) {
                sparse_mat.coeffRef(i0 + i, j0 + j) = dense_mat(i, j);
            } else {
                sparse_mat.insert(i0 + i, j0 + j)   = dense_mat(i, j);
            }
        }
    }
}

void assign_diagonal(Eigen::Index i0, Eigen::Index i1,
                     const VectorXs& diag_vec,
                     Eigen::SparseMatrix<scalar>& sparse_mat,
                     bool update) {
    assert(i0 <= i1);
    assert((i1 - i0) == diag_vec.size());
    assert(i1 <= sparse_mat.rows());
    assert(i1 <= sparse_mat.cols());
    for (Eigen::Index i = i0; i < i1; ++i) {
        if (update) {
            sparse_mat.coeffRef(i, i) = diag_vec(i - i0);
        } else {
            sparse_mat.insert(i, i)   = diag_vec(i - i0);
        }
    }
}

void assign_diagonal(Eigen::Index i0, Eigen::Index i1,
                     scalar value,
                     Eigen::SparseMatrix<scalar>& sparse_mat,
                     bool update) {
    assert(i0 <= i1);
    assert(i1 <= sparse_mat.rows());
    assert(i1 <= sparse_mat.cols());
    for (Eigen::Index i = i0; i < i1; ++i) {
        if (update) {
            sparse_mat.coeffRef(i, i) = value;
        } else {
            sparse_mat.insert(i, i)   = value;
        }
    }
}

class KKTSystem {
public:
    KKTSystem(int nx, int nu, int N, std::vector<int>& ncs) {
        // int nxu = nx + nu;   
        // max_num_triplets_ = 0;
        // for (int k = 0; k < N; ++k) {
        //     int nc = ncs[k];
        //     if (k == 0) {
        //         /*
        //         ** [R0  B0'         Du0'          0
        //         **  B0 -rho_eq * I  0            -I
        //         **  Du0 0          -rho_ineq * I  0]
        //         */
        //         max_num_triplets_ += nu * nu; // R
        //         max_num_triplets_ += nx * nu; // B
        //         max_num_triplets_ += nc * nu; // Du
        //         max_num_triplets_ += nx + nc; // regularization
        //         max_num_triplets_ += nx; // -I_{nx}
        //     } else {
        //         /**
        //         ** [-I 0 Qk  Sk'  Ak'         Dxk'          0
        //         **   0 0 Sk  Rk'  Bk'         Duk'          0
        //         **   0 0 Ak  Bk  -rho_eq * I  0            -I
        //         **   0 0 Dxk Duk  0          -rho_ineq * I  0]
        //         */
        //         max_num_triplets_ += nxu * nxu; // Q, S, R
        //         max_num_triplets_ += nx * nxu;  // A, B
        //         max_num_triplets_ += nc * nxu;  // Dx, Du
        //         max_num_triplets_ += nxu + nc; // regularization
        //         max_num_triplets_ += nx; // -I_{nx}
        //     }
        // }
        // // terminal stage
        // /*
        // ** [-I 0 QN   DxN'
        // **   0 0 DxN -rho_ineq * I]
        // */
        // max_num_triplets_ += nx * nx; // P
        // max_num_triplets_ += nc * nx; // Dx
        // max_num_triplets_ += nc;      // regularization  

        // triplets_.reserve(max_num_triplets_);

        int nxu = nx + nu;
        int num_rows = 0;
        num_rows += nxu + ncs[0]; // stage 0
        for (int k = 1; k < N; ++k) {
            num_rows += nxu + ncs[k] + nx; // stage k
        }
        num_rows += nx + ncs[N]; // terminal stage
        KKT_mat_.resize(num_rows, num_rows);
        
        rhs_.resize(num_rows);
    }
    
    ~KKTSystemBuilder();

    void form_KKT_system(const LQRModel& model, 
                            const scalar rho_eq, 
                            const std::vector<scalar>& rho_ineq_vec,
                            bool update) 
    {
        /*
        ** [P   G'
        **  G  -diag(rho)]
        */
        if (!update) {
            KKT_mat_.setZero();
        }
        rhs_.setZero();
        
        int nx = model.n;
        int nu = model.m;
        int nxu = nx + nu;
        int N  = model.N;
        int row_offset = 0;
        
        // Stage 0: [R0  B0'         Du0'          0  ]
        //          [B0 -rho_eq * I  0            -I  ]
        //          [Du0 0          -rho_ineq * I  0  ]
        {
            const auto& kpoint = model.get_knotpoint(0);
            int nc = kpoint.get_constraint_dim();
            
            // Cost matrix R0 for stage 0
            const auto& R0 = kpoint.H.topLeftCorner(nu, nu);
            assign_dense_matrix(row_offset, row_offset, R0, KKT_mat_, update);
            
            // Dynamics constraint: B0 and -rho_eq * I
           
            // B matrix (control to next state)
            const auto& B = kpoint.E.leftCols(nu);
            assign_dense_matrix(row_offset, row_offset + nu, B.transpose(), KKT_mat_, update);
            
            // -rho_eq * I for dynamics regularization
            assign_diagonal(row_offset + nu, row_offset + nu + nx, -rho_eq, KKT_mat_, update);
            
            // Inequality constraints: Du0 and -rho_ineq * I
            if (nc > 0) {
                const auto& Du0 = kpoint.D_con.leftCols(nu);
                assign_dense_matrix(row_offset, row_offset + nxu, Du0.transpose(), KKT_mat_, update);
                
                // -rho_ineq * I for inequality regularization
                VectorXs neg_rho_ineq_vec = VectorXs::Constant(nc, -rho_ineq_vec[0]);
                assign_diagonal(row_offset + nxu, row_offset + nxu + nc, 
                               neg_rho_ineq_vec, KKT_mat_, update);
            }
            
            // -I coupling to next stage
            VectorXs neg_ones = VectorXs::Constant(nx, -1.0);
            assign_diagonal(row_offset + nxu, row_offset + nxu + nx + nc, neg_ones, KKT_mat_, update);
            
            row_offset += nxu + nx + nc;
        }
        
        // Intermediate stages k = 1, ..., N-1
        for (int k = 1; k < N; ++k) {
            const auto& kpoint = model.knotpoints[k];
            int nc = kpoint.n_con;
            
            // Structure: [-I 0 Qk  Sk'  Ak'         Dxk'          0]
            //            [ 0 0 Sk  Rk'  Bk'         Duk'          0]
            //            [ 0 0 Ak  Bk  -rho_eq * I  0            -I]
            //            [ 0 0 Dxk Duk  0          -rho_ineq * I  0]
            
            // Previous stage coupling: -I block
            VectorXs neg_ones = VectorXs::Constant(nx, -1.0);
            assign_diagonal(row_offset, row_offset, neg_ones, KKT_mat_, update);
            
            // Cost matrix H = [R S'; S Q] for current stage
            assign_dense_matrix(row_offset + nx, row_offset + nx, kpoint.H, KKT_mat_, update);
            
            // Dynamics constraint: A, B matrices and regularization
            if (kpoint.E.size() > 0) {
                MatrixXs B = kpoint.E.leftCols(nu);
                MatrixXs A = kpoint.E.rightCols(nx);
                
                // A matrix (state transition)
                assign_dense_matrix(row_offset + nxu + nx, row_offset + nx + nu, A, KKT_mat_, update);
                assign_dense_matrix(row_offset + nx + nu, row_offset + nxu + nx, A.transpose(), KKT_mat_, update);
                
                // B matrix (control input)
                assign_dense_matrix(row_offset + nxu + nx, row_offset + nx, B, KKT_mat_, update);
                assign_dense_matrix(row_offset + nx, row_offset + nxu + nx, B.transpose(), KKT_mat_, update);
                
                // -rho_eq * I for dynamics regularization
                VectorXs neg_rho_eq_vec = VectorXs::Constant(nx, -rho_eq);
                assign_diagonal(row_offset + nxu + nx, row_offset + nxu + nx + nx, 
                               neg_rho_eq_vec, KKT_mat_, update);
            }
            
            // Inequality constraints
            if (nc > 0) {
                MatrixXs Du = kpoint.D_con.leftCols(nu);
                MatrixXs Dx = kpoint.D_con.rightCols(nx);
                
                // Du matrix
                assign_dense_matrix(row_offset + nxu + 2*nx, row_offset + nx, Du, KKT_mat_, update);
                assign_dense_matrix(row_offset + nx, row_offset + nxu + 2*nx, Du.transpose(), KKT_mat_, update);
                
                // Dx matrix
                assign_dense_matrix(row_offset + nxu + 2*nx, row_offset + nx + nu, Dx, KKT_mat_, update);
                assign_dense_matrix(row_offset + nx + nu, row_offset + nxu + 2*nx, Dx.transpose(), KKT_mat_, update);
                
                // -rho_ineq * I for inequality regularization
                VectorXs neg_rho_ineq_vec = VectorXs::Constant(nc, -rho_ineq_vec[k]);
                assign_diagonal(row_offset + nxu + 2*nx, row_offset + nxu + 2*nx + nc, 
                               neg_rho_ineq_vec, KKT_mat_, update);
            }
            
            // -I coupling to next stage
            assign_diagonal(row_offset + nxu + nx, row_offset + nxu + 2*nx + nc, neg_ones, KKT_mat_, update);
            
            row_offset += nxu + 2*nx + nc;
        }
        
        // Terminal stage: [-I 0 QN   DxN'          ]
        //                 [ 0 0 DxN -rho_ineq * I ]
        {
            const auto& terminal_kpoint = model.knotpoints[N];
            int nc = terminal_kpoint.n_con;
            
            // Previous stage coupling: -I block
            VectorXs neg_ones = VectorXs::Constant(nx, -1.0);
            assign_diagonal(row_offset, row_offset, neg_ones, KKT_mat_, update);
            
            // Terminal cost matrix QN
            assign_dense_matrix(row_offset + nx, row_offset + nx, terminal_kpoint.H, KKT_mat_, update);
            
            // Terminal constraints
            if (nc > 0) {
                MatrixXs DxN = terminal_kpoint.D_con;
                assign_dense_matrix(row_offset + 2*nx, row_offset + nx, DxN, KKT_mat_, update);
                assign_dense_matrix(row_offset + nx, row_offset + 2*nx, DxN.transpose(), KKT_mat_, update);
                
                // -rho_ineq * I for terminal inequality regularization
                VectorXs neg_rho_ineq_vec = VectorXs::Constant(nc, -rho_ineq_vec[N]);
                assign_diagonal(row_offset + 2*nx, row_offset + 2*nx + nc, 
                               neg_rho_ineq_vec, KKT_mat_, update);
            }
        }
    }

private:
    // std::vector<Eigen::Triplet<scalar>> triplets_;
    // int max_num_triplets_;
    Eigen::SparseMatrix<scalar> KKT_mat_;
    VectorXs rhs_;
}; 

} // namespace lqr