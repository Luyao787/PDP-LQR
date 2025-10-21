#include <iostream>
#include <Eigen/Dense>
#include "clqr/typedefs.hpp"
#include "clqr/lqr_model.hpp"
#include "clqr/results.hpp"
#include "clqr/settings.hpp"
#include "clqr/osclqr_solver.hpp"
#include "clqr/lqr/qdldl_solver.hpp"

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
        int nc = k > 0 ? nx + nu : nu;
        // int nc = 0;
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
        // kpoint.D_con.topLeftCorner(nu, nu).setIdentity();
        // if (k > 0) {
        //     kpoint.D_con.bottomRightCorner(nx, nx).setIdentity();
        // }
        if (k == 0) {
            kpoint.D_con.topLeftCorner(nu, nu).setIdentity();
            kpoint.e_lb = u_min;
            kpoint.e_ub = u_max;
        } else {
            kpoint.D_con.topLeftCorner(nu, nu).setIdentity();
            kpoint.D_con.bottomRightCorner(nx, nx).setIdentity();
            kpoint.e_lb << u_min, x_min;
            kpoint.e_ub << u_max, x_max;
        }         
    }
    int nc = nx;
    // int nc = 0;
    ncs[N] = nc;
    lqr_model.add_node(nx, nu, nc, N, true);
    auto& kpoint = lqr_model.nodes[N];
    kpoint.H = Q;
    kpoint.h = q;
    kpoint.D_con.setIdentity();
    kpoint.e_lb = x_min;
    kpoint.e_ub = x_max;

    std::vector<VectorXs> ws, ys, zs, rho_vecs, inv_rho_vecs;
    scalar rho = 0.01;
    scalar sigma = 1e-6;
    ws.resize(N + 1);
    ys.resize(N + 1);
    zs.resize(N + 1);
    rho_vecs.resize(N + 1);
    inv_rho_vecs.resize(N + 1);

    // Initialize each vector element
    for (int k = 0; k < N + 1; ++k) {
        if (k < N) {
            ws[k].resize(nx + nu);
        } else {
            ws[k].resize(nx);
        }
        ws[k].setZero();
        ys[k].resize(lqr_model.ncs[k]);
        ys[k].setZero();
        zs[k].resize(lqr_model.ncs[k]);
        zs[k].setZero();
        rho_vecs[k].resize(lqr_model.ncs[k]);
        rho_vecs[k].setConstant(rho);
        inv_rho_vecs[k].resize(lqr_model.ncs[k]);
        inv_rho_vecs[k].setConstant(1.0 / rho);
    }

    QDLDLSolver lqr_solver(lqr_model);
    lqr_solver.update_problem_data(ws, ys, zs, inv_rho_vecs, sigma);
    lqr_solver.backward(inv_rho_vecs);
    lqr_solver.forward(x0, ws);
    std::cout << "First input:\n" << ws[0].head(nu).transpose() << std::endl;
    std::cout << "Final state:\n" << ws[N].tail(nx).transpose() << std::endl;


    // Initialize each vector element
    for (int k = 0; k < N + 1; ++k) {
        if (k < N) {
            ws[k].resize(nx + nu);
        } else {
            ws[k].resize(nx);
        }
        ws[k].setZero();
        ys[k].resize(lqr_model.ncs[k]);
        ys[k].setZero();
        zs[k].resize(lqr_model.ncs[k]);
        zs[k].setZero();
        rho_vecs[k].resize(lqr_model.ncs[k]);
        rho_vecs[k].setConstant(rho);
        inv_rho_vecs[k].resize(lqr_model.ncs[k]);
        inv_rho_vecs[k].setConstant(1.0 / rho);
    }

    LQRSolver lqr_solver2(lqr_model);
    lqr_solver2.update_problem_data(ws, ys, zs, inv_rho_vecs, sigma);
    lqr_solver2.backward(rho_vecs);
    lqr_solver2.forward(x0, ws);
    std::cout << "First input (LQRSolver):\n" << ws[0].head(nu).transpose() << std::endl;
    std::cout << "Final state (LQRSolver):\n" << ws[N].tail(nx).transpose() << std::endl;


    // Initialize each vector element
    for (int k = 0; k < N + 1; ++k) {
        if (k < N) {
            ws[k].resize(nx + nu);
        } else {
            ws[k].resize(nx);
        }
        ws[k].setZero();
        ys[k].resize(lqr_model.ncs[k]);
        ys[k].setZero();
        zs[k].resize(lqr_model.ncs[k]);
        zs[k].setZero();
        rho_vecs[k].resize(lqr_model.ncs[k]);
        rho_vecs[k].setConstant(rho);
        inv_rho_vecs[k].resize(lqr_model.ncs[k]);
        inv_rho_vecs[k].setConstant(1.0 / rho);
    }

    LQRParallelSolver lqr_solver3(lqr_model, 2, true);
    lqr_solver3.update_problem_data(ws, ys, zs, inv_rho_vecs, sigma);
    lqr_solver3.backward(rho_vecs);
    lqr_solver3.forward(x0, ws);
    std::cout << "First input (LQRParallelSolver):\n" << ws[0].head(nu).transpose() << std::endl;
    std::cout << "Final state (LQRParallelSolver):\n" << ws[N].tail(nx).transpose() << std::endl;

}