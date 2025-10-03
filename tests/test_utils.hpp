#pragma once

#include <memory>
#include <vector>
#include <lqr/lqr.hpp>

using namespace lqr;

/**
 * @brief Create a LQR model for quadrotor system
 * 
 * Creates a 12-state, 4-control quadrotor LQR model with appropriate
 * dynamics matrices, cost functions, and constraints for testing.
 * 
 * @return std::unique_ptr<lqr::LQRModel> Configured LQR model
 */
std::unique_ptr<lqr::LQRModel> create_LQR_model() {
    const int nx = 12; // State dimension
    const int nu = 4;  // Control dimension
    const int N = 10;  // Horizon length

    VectorXs x_ref(nx);
    x_ref << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
    
    VectorXs x_min(nx), x_max(nx), u_min(nu), u_max(nu);
    x_min << -0.52359878, -0.52359878, -LQR_INFTY, 
                -LQR_INFTY, -LQR_INFTY, -1.,
                -LQR_INFTY, -LQR_INFTY, -LQR_INFTY, 
                -LQR_INFTY, -LQR_INFTY, -LQR_INFTY;
    x_max << 0.52359878, 0.52359878, LQR_INFTY, 
                LQR_INFTY, LQR_INFTY, LQR_INFTY,
                LQR_INFTY, LQR_INFTY, 2.5, 
                LQR_INFTY, LQR_INFTY, LQR_INFTY;
    u_min << -0.9916, -0.9916, -0.9916, -0.9916;
    u_max << 2.4084, 2.4084, 2.4084, 2.4084;

    MatrixXs A(nx, nx);
    MatrixXs B(nx, nu);

    A << 
    1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.,
    0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.,
    0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.,
    0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.,
    0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.,
    0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992,
    0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.,
    0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.,
    0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.,
    0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.,
    0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.,
    0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846;

    B <<
    0.,      -0.0726,  0.,     0.0726,
    -0.0726,   0.,      0.0726, 0.,
    -0.0152,   0.0152, -0.0152, 0.0152,
    -0.,      -0.0006, -0.,     0.0006,
    0.0006,   0.,     -0.0006, 0.0000,
    0.0106,   0.0106,  0.0106, 0.0106,
    0.,      -1.4512,  0.,     1.4512,
    -1.4512,   0.,      1.4512, 0.,
    -0.3049,   0.3049, -0.3049, 0.3049,
    -0.,      -0.0236,  0.,     0.0236,
    0.0236,   0.,     -0.0236, 0.,
    0.2107,   0.2107,  0.2107, 0.2107;
    
    MatrixXs Q(nx, nx);
    MatrixXs R(nu, nu);
    MatrixXs S(nu, nx);
    VectorXs q(nx);
    VectorXs r(nu);

    Q.diagonal() << 0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.;
    R.diagonal() << 0.1, 0.1, 0.1, 0.1;
    S.setZero();
    q = -x_ref.transpose() * Q;
    r.setZero();

    auto lqr_model = std::make_unique<LQRModel>(nx, nu, N);
    std::vector<int> ncs(N + 1); 

    for (int k = 0; k < N; ++k) {
        int nc = nu + nx;
        ncs[k] = nc;
        lqr_model->add_knotpoint(nx, nu, nc, k);
        auto& kpoint = lqr_model->knotpoints[k];

        kpoint.E << B, A;
        VectorXs c = VectorXs::Zero(nx); // Add missing variable declaration
        kpoint.c = c;

        kpoint.H.block(0, 0, nu, nu) = R;
        kpoint.H.block(nu, nu, nx, nx) = Q;
        kpoint.H.block(0, nu, nu, nx) = S;
        kpoint.H.block(nu, 0, nx, nu) = S.transpose();       
        kpoint.h.head(nu) = r;
        kpoint.h.tail(nx) = q;

        kpoint.D_con.setZero();
        kpoint.D_con.topLeftCorner(nu, nu).setIdentity();
        if (k > 0) {
            kpoint.D_con.bottomRightCorner(nx, nx).setIdentity();
        }
        kpoint.e_lb << u_min, x_min;
        kpoint.e_ub << u_max, x_max;          
    }
    int nc = nx;
    ncs[N] = nc;
    lqr_model->add_knotpoint(nx, nu, nc, N, true);
    auto& kpoint = lqr_model->knotpoints[N];
    kpoint.H = Q;
    kpoint.h = q;
    kpoint.D_con.setIdentity();
    kpoint.e_lb = x_min;
    kpoint.e_ub = x_max;

    return lqr_model;
}