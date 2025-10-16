#include <iostream>
#include <Eigen/Dense>
#include "clqr/typedefs.hpp"
#include "clqr/lqr_model.hpp"
#include "clqr/results.hpp"
#include "clqr/settings.hpp"
#include "clqr/osclqr_solver.hpp"

using namespace lqr;

int main() {
    std::cout << "PDP-LQR Library Example" << std::endl;
    
    /*
    * Example: A quadrotor example adapted from https://osqp.org/docs/release-0.6.3/examples/mpc.html
    */

    constexpr int nx = 12;
    constexpr int nu = 4;
    constexpr int N  = 10;

    VectorXs x0(nx);
    VectorXs x_ref(nx);
    x0 << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
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
    VectorXs c(nx);

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
    
    c.setZero();

    MatrixXs Q(nx, nx);
    MatrixXs R(nu, nu);
    MatrixXs S(nu, nx);
    VectorXs q(nx);
    VectorXs r(nu);

    Q.setZero();
    Q.diagonal() << 0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.;
    R.setZero();    
    R.diagonal() << 0.1, 0.1, 0.1, 0.1;
    S.setZero();
    q = -x_ref.transpose() * Q;
    r.setZero();
    
    LQRModel lqr_model(nx, nu, N);

    std::vector<int> ncs(N + 1);
    for (int k = 0; k < N; ++k) {
        int nc = nu + nx;
        ncs[k] = nc;
        lqr_model.add_node(nx, nu, nc, k);
        auto& kpoint = lqr_model.nodes[k];

        kpoint.E << B, A;
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
    lqr_model.add_node(nx, nu, nc, N, true);
    auto& kpoint = lqr_model.nodes[N];
    kpoint.H = Q;
    kpoint.h = q;
    kpoint.D_con.setIdentity();
    kpoint.e_lb = x_min;
    kpoint.e_ub = x_max;

    LQRResults lqr_results;
    lqr_results.reset(nx, nu, ncs, N);

    LQRSettings lqr_settings;
    lqr_settings.max_iter = 1000;

    OSCLQRSolver lqr_solver(lqr_model);
    lqr_solver.solve(x0, lqr_settings, lqr_results);

    std::cout << "Final state: " << lqr_results.xs[N].transpose() << std::endl;
    // Control inputs
    for (int k = 0; k < std::min(5, N); ++k) {
        std::cout << "Control input at step " << k << ": " << lqr_results.us[k].transpose() << std::endl;
    }

}