#include <lqr/lqr.hpp>
#include <iostream>

using namespace lqr;

int main() {
    std::cout << "PDP-LQR Library Example" << std::endl;
    
    constexpr int nx = 12;
    constexpr int nu = 4;
    constexpr int N  = 10;

    VectorXs x0(nx);
    VectorXs x_ref(nx);
    x0 << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
    x_ref << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.;

    VectorXs x_min(nx), x_max(nx), u_min(nu), u_max(nu);
    x_min << -0.52359878, -0.52359878, -LQR_INFTY, -LQR_INFTY, -LQR_INFTY, -1.,
        -LQR_INFTY, -LQR_INFTY, -LQR_INFTY, -LQR_INFTY, -LQR_INFTY, -LQR_INFTY;
    x_max << 0.52359878, 0.52359878, LQR_INFTY, LQR_INFTY, LQR_INFTY, LQR_INFTY,
        LQR_INFTY, LQR_INFTY, 2.5, LQR_INFTY, LQR_INFTY, LQR_INFTY;
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

    LQRModel lqr_model;
    LQRSettings lqr_settings;
    lqr_settings.max_iter = 1000;
    
    lqr_model.n = nx;
    lqr_model.m = nu;
    lqr_model.N = N; 

    std::vector<int> ncs(N + 1);
    for (int k = 0; k < N; ++k) {
        int nc = nu + nx;
        ncs[k] = nc;
        lqr_model.knotpoints.emplace_back(nx, nu, nc, k);
        lqr_model.knotpoints[k].E << B, A;
        lqr_model.knotpoints[k].c = c;
        
        lqr_model.knotpoints[k].H.block(0, 0, nu, nu) = R;
        lqr_model.knotpoints[k].H.block(nu, nu, nx, nx) = Q;
        lqr_model.knotpoints[k].H.block(0, nu, nu, nx) = S;
        lqr_model.knotpoints[k].H.block(nu, 0, nx, nu) = S.transpose();       
        lqr_model.knotpoints[k].h.head(nu) = r;
        lqr_model.knotpoints[k].h.tail(nx) = q;

        lqr_model.knotpoints[k].D_con.setZero();
        lqr_model.knotpoints[k].D_con.topLeftCorner(nu, nu).setIdentity();
        if (k > 0) {
            lqr_model.knotpoints[k].D_con.bottomRightCorner(nx, nx).setIdentity();
        }
        lqr_model.knotpoints[k].e_lb << u_min, x_min;
        lqr_model.knotpoints[k].e_ub << u_max, x_max;          
    }
    int nc = nx;
    ncs[N] = nc;
    lqr_model.knotpoints.emplace_back(nx, nu, nc, N, true);
    lqr_model.knotpoints[N].H = Q;
    lqr_model.knotpoints[N].h = q;
    lqr_model.knotpoints[N].D_con.setIdentity();
    lqr_model.knotpoints[N].e_lb = x_min;
    lqr_model.knotpoints[N].e_ub = x_max;
    
    LQRResults lqr_results;
    lqr_results.reset(nx, nu, ncs, N);

    SerialCLQRSolver clqr_solver(lqr_model);

    // ParallelCLQRSolver clqr_solver(lqr_model, 1);
    // ParallelCLQRSolver clqr_solver(lqr_model, 2);

    clqr_solver.solve(x0, lqr_settings, lqr_results);

    auto tic0 = omp_get_wtime();
    clqr_solver.clear_workspace();
    clqr_solver.solve(x0, lqr_settings, lqr_results);
    auto toc1 = omp_get_wtime();
    std::cout << "Time for second solve call: " << (toc1 - tic0) * 1e3 << " ms" << "\n" << std::endl;

    // std::cout << "Final state: " << lqr_results.xs[N].transpose() << std::endl;
    // // Control inputs
    // for (int k = 0; k < N; ++k) {
    //     std::cout << "Control input at step " << k << ": " << lqr_results.us[k].transpose() << std::endl;
    // }

    // for (int k = 0; k <= N; ++k) {
    //     std::cout << "State at step " << k << ": " << lqr_results.xs[k].transpose() << std::endl;
    // }

    // std::cout << "Dual variables (lambdas):" << std::endl;
    // for (int k = 0; k <= N; ++k) {
    //     std::cout << "Lambda at step " << k << ": " << lqr_results.lambdas[k].transpose() << std::endl;
    // } 
}