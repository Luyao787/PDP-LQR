#pragma once

#include <qdldl/qdldl.h>
#include <Eigen/Sparse>
#include "clqr/lqr_model.hpp"
#include "qdldl_typedefs.hpp"

namespace lqr {

void assign_dense_matrix(Eigen::Index i0, Eigen::Index j0,
                         const MatrixXs& dense_mat,
                         Eigen::SparseMatrix<scalar>& sparse_mat,
                         bool update, bool fill_upper = false, bool ignore_zeros = false) {
    assert(i0 + dense_mat.rows() <= sparse_mat.rows());
    assert(j0 + dense_mat.cols() <= sparse_mat.cols());

    for (Eigen::Index j = 0; j < dense_mat.cols(); ++j) {
        for (Eigen::Index i = 0; i < dense_mat.rows(); ++i) {
            if (!fill_upper || i <= j) {
                if (update) {
                    if (ignore_zeros && dense_mat(i, j) == scalar(0)) {
                        continue;
                    }
                    sparse_mat.coeffRef(i0 + i, j0 + j) = dense_mat(i, j);
                } else {
                    if (ignore_zeros && dense_mat(i, j) == scalar(0)) {
                        continue;
                    }
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
                            scalar value, int size,
                            Eigen::SparseMatrix<scalar>& sparse_mat,
                            bool update) {
    assert(i0 + size <= sparse_mat.rows());
    assert(j0 + size <= sparse_mat.cols());
    for (int i = 0; i < size; ++i) {
        if (update) {
            sparse_mat.coeffRef(i0 + i, j0 + i) = value;
        } else {
            sparse_mat.insert(i0 + i, j0 + i)   = value;
        }
    }
}

} // namespace lqr