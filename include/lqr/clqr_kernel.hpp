#pragma once

#include "lqr/lqr_model.hpp"
#include "lqr/typedefs.hpp"

namespace lqr {

struct CLQRKernelData {
    MatrixXs H; 
    VectorXs h;

    /* Variables used during backward pass */
    MatrixXs L;    // L = [Luu Lux; Lxu Lxx]
    MatrixXs V, M; // V = E^T * Lxx, M = H + V * V^T

    VectorXs lp;  
    VectorXs Pb, Pb_tmp;

    /* Constraints */
    VectorXs z, z_tilde, z_prev, y;
    
    MatrixXs rhoD_tmp;
    VectorXs Dw_tmp, y_tmp, Hw_tmp, Dty_tmp;

    VectorXs prim_res, dual_res;

    VectorXs rho_vec, inv_rho_vec;
    /* ----------------------------------------- */

    VectorXs w;
    VectorXs lambda;

    bool is_terminal;

    CLQRKernelData(int nx, int nu, int nc, bool is_terminal_stage=false):
        is_terminal(is_terminal_stage)
    {
        int nxu = nx + nu;
        if (!is_terminal) {
            H.resize(nxu, nxu);
            h.resize(nxu);
            L.resize(nxu, nxu);
            lp.resize(nxu);
            w.resize(nxu);
        }
        else {
            H.resize(nx, nx);
            h.resize(nx);
            L.resize(nx, nx);
            lp.resize(nx);
            w.resize(nx);
        }
        
        V.resize(nxu, nx);
        M.resize(nxu, nxu);
        Pb.resize(nx);
        Pb_tmp.resize(nx);

        lambda.resize(nx);

        if (nc > 0) {
            if (!is_terminal) {
                rhoD_tmp.resize(nc, nxu);
                Dty_tmp.resize(nxu);
                Hw_tmp.resize(nxu);
                dual_res.resize(nxu);
            }
            else 
            {
                rhoD_tmp.resize(nc, nx);
                Dty_tmp.resize(nx);
                Hw_tmp.resize(nx);
                dual_res.resize(nx);
            }
            z.resize(nc);
            z_tilde.resize(nc);
            z_prev.resize(nc);
            y.resize(nc);
            rho_vec.resize(nc);
            inv_rho_vec.resize(nc);
            Dw_tmp.resize(nc);
            y_tmp.resize(nc);
            prim_res.resize(nc);
        }
        set_zero();
    }

    void set_zero() {
        H.setZero();
        h.setZero();
        L.setZero();
        lp.setZero();
        w.setZero();
        lambda.setZero();

        V.setZero();
        M.setZero();
        Pb.setZero();
        Pb_tmp.setZero();

        if (rho_vec.size() > 0) {
            rho_vec.setZero();
            inv_rho_vec.setZero();
            z.setZero();
            z_tilde.setZero();
            y.setZero();
            rhoD_tmp.setZero();
            Dw_tmp.setZero();
            y_tmp.setZero();
            Hw_tmp.setZero();
            Dty_tmp.setZero();
            prim_res.setZero();
            dual_res.setZero();
        }
    }
};

struct CLQRKernel {
    
    inline static void terminal_step_with_factorization(const Knotpoint& kpoint, CLQRKernelData& data, scalar sigma) {
        data.H = kpoint.H;
        data.H.diagonal().array() += sigma;
        data.h = kpoint.h;
        data.h.noalias() -= sigma * data.w;

        if (kpoint.n_con > 0) {
            data.rhoD_tmp.noalias() = data.rho_vec.asDiagonal() * kpoint.D_con;
            data.H.noalias() += kpoint.D_con.transpose() * data.rhoD_tmp;

            data.y_tmp = data.y;
            data.y_tmp.noalias() -= data.rho_vec.cwiseProduct(data.z);
            data.h.noalias() += kpoint.D_con.transpose() * data.y_tmp;
        }

        data.L  = data.H.llt().matrixL();
        data.lp = data.h;
    }

    inline static void terminal_step_without_factorization(const Knotpoint& kpoint, CLQRKernelData& data, scalar sigma) {
        data.h = kpoint.h;
        data.h.noalias() -= sigma * data.w;

        if (kpoint.n_con > 0) {
            data.y_tmp = data.y;
            data.y_tmp.noalias() -= data.rho_vec.cwiseProduct(data.z);
            data.h.noalias() += kpoint.D_con.transpose() * data.y_tmp;
        }

        data.lp = data.h;
    }

    inline static void step_with_factorization(const Knotpoint& kpoint, CLQRKernelData& data, const CLQRKernelData& data_next, scalar sigma) {
        data.H = kpoint.H;
        data.H.diagonal().array() += sigma;
        data.h = kpoint.h;
        data.h.noalias() -= sigma * data.w;
        
        if (kpoint.n_con > 0) {
            data.rhoD_tmp.noalias() = data.rho_vec.asDiagonal() * kpoint.D_con;
            data.H.noalias() += kpoint.D_con.transpose() * data.rhoD_tmp;

            data.y_tmp = data.y;
            data.y_tmp.noalias() -= data.rho_vec.cwiseProduct(data.z);
            data.h.noalias() += kpoint.D_con.transpose() * data.y_tmp;
        }
        
        MatrixRef V_k      = data.V;
        MatrixRef M_k      = data.M;
        MatrixRef L_k      = data.L;

        ConstMatrixRef E_k = kpoint.E;
        ConstMatrixRef Lxx_next = data_next.L.bottomRightCorner(kpoint.n, kpoint.n);   

        V_k.noalias() = E_k.transpose() * Lxx_next;

        M_k = data.H;
        M_k.noalias() += V_k * V_k.transpose();
        
        L_k = M_k.llt().matrixL();

        ConstVectorRef p_next = data_next.lp.tail(kpoint.n);
        ConstMatrixRef Luu_k  = L_k.topLeftCorner(kpoint.m, kpoint.m);
        ConstMatrixRef Lxu_k  = L_k.bottomLeftCorner(kpoint.n, kpoint.m);
        
        VectorRef Pb_k   = data.Pb;
        VectorRef Pb_tmp = data.Pb_tmp;
        VectorRef lp_k   = data.lp;
        VectorRef lu_k   = lp_k.head(kpoint.m);
        VectorRef p_k    = lp_k.tail(kpoint.n);

        Pb_tmp.noalias() = Lxx_next.transpose() * kpoint.c;
        Pb_k.noalias()   = Lxx_next * Pb_tmp;
        Pb_k            += p_next;

        lp_k             = data.h;
        lp_k.noalias()  += E_k.transpose() * Pb_k;

        Luu_k.triangularView<Eigen::Lower>().solveInPlace(lu_k);
        p_k.noalias()   -= Lxu_k * lu_k;
    }

    inline static void step_without_factorization(const Knotpoint& kpoint, CLQRKernelData& data, const CLQRKernelData& data_next, scalar sigma) {
        data.h = kpoint.h;
        data.h.noalias() -= sigma * data.w;

        if (kpoint.n_con > 0) {
            data.y_tmp = data.y;
            data.y_tmp.noalias() -= data.rho_vec.cwiseProduct(data.z);
            data.h.noalias() += kpoint.D_con.transpose() * data.y_tmp;
        }

        ConstMatrixRef E_k      = kpoint.E;
        ConstMatrixRef Lxx_next = data_next.L.bottomRightCorner(kpoint.n, kpoint.n);   
        ConstMatrixRef Luu_k    = data.L.topLeftCorner(kpoint.m, kpoint.m);
        ConstMatrixRef Lxu_k    = data.L.bottomLeftCorner(kpoint.n, kpoint.m);
        ConstVectorRef p_next   = data_next.lp.tail(kpoint.n);
        
        VectorRef Pb_k   = data.Pb;
        VectorRef Pb_tmp = data.Pb_tmp;
        VectorRef lp_k   = data.lp;
        VectorRef lu_k   = lp_k.head(kpoint.m);
        VectorRef p_k    = lp_k.tail(kpoint.n);

        Pb_tmp.noalias() = Lxx_next.transpose() * kpoint.c;
        Pb_k.noalias()   = Lxx_next * Pb_tmp;
        Pb_k            += p_next;

        lp_k             = data.h;
        lp_k.noalias()  += E_k.transpose() * Pb_k;

        Luu_k.triangularView<Eigen::Lower>().solveInPlace(lu_k);
        p_k.noalias()   -= Lxu_k * lu_k;
    }

    inline static void forward_step(const Knotpoint& kpoint, CLQRKernelData& data, CLQRKernelData& data_next) {
        int nx = kpoint.n;
        int nu = kpoint.m;
        
        ConstMatrixRef A   = kpoint.E.rightCols(nx); 
        ConstMatrixRef B   = kpoint.E.leftCols(nu);   
        ConstVectorRef c   = kpoint.c;
        MatrixRef      Luu = data.L.topLeftCorner(nu, nu);
        ConstMatrixRef Lxu = data.L.bottomLeftCorner(nx, nu);
        ConstVectorRef lu  = data.lp.head(nu);

        VectorRef x      = data.w.tail(nx);
        VectorRef u      = data.w.head(nu);
        VectorRef x_next = data_next.w.tail(nx);
        
        u = -lu;
        u.noalias() -= Lxu.transpose() * x;
        Luu.triangularView<Eigen::Lower>().transpose().solveInPlace(u);

        x_next            = c;
        x_next.noalias() += A * x;
        x_next.noalias() += B * u;

        ConstMatrixRef Lxx_next = data_next.L.bottomRightCorner(nx, nx);
        ConstVectorRef p_next   = data_next.lp.tail(nx);
        VectorRef lambda_next   = data_next.lambda;

        lambda_next.noalias() = Lxx_next.transpose() * x_next;
        lambda_next           = Lxx_next * lambda_next; // more efficient?
        lambda_next          += p_next;
    }

    inline static void residual_step(const Knotpoint& kpoint, 
                                     CLQRKernelData& data, CLQRKernelData& data_next,
                                     scalar& prim_res_norm, scalar& dual_res_norm,
                                     scalar& prim_res_norm_rel, scalar& dual_res_norm_rel) {
        // int nx = kpoint.n;
        if (kpoint.n_con > 0) {
            data.prim_res = data.Dw_tmp - data.z;

            // data.Hw_tmp.noalias()    = kpoint.H * data.w;
            // data.Dty_tmp.noalias()   = kpoint.D_con.transpose() * data.y;

            // data.dual_res            = data.Hw_tmp;
            // data.dual_res.noalias() += kpoint.h;
            // data.dual_res.noalias() += data.Dty_tmp;

            // data.dual_res.noalias() += kpoint.E.transpose() * data_next.lambda;
            // data.dual_res.tail(nx)  -= data.lambda;

            // if (kpoint.time_step == 0) {
            //     data.dual_res.tail(nx).setZero();
            // }
            
            data.Dty_tmp.noalias()   = kpoint.D_con.transpose() * data.y;
            data.dual_res = kpoint.D_con * data.rho_vec.cwiseProduct(data.z - data.z_prev);

            prim_res_norm_rel = std::max(
                prim_res_norm_rel, data.Dw_tmp.lpNorm<Eigen::Infinity>());
            prim_res_norm_rel = std::max(
                prim_res_norm_rel, data.z.lpNorm<Eigen::Infinity>());
            
            // dual_res_norm_rel = std::max(
            //     dual_res_norm_rel, data.Hw_tmp.lpNorm<Eigen::Infinity>());
            // dual_res_norm_rel = std::max(
            //     dual_res_norm_rel, kpoint.h.lpNorm<Eigen::Infinity>());
            // dual_res_norm_rel = std::max(
            //     dual_res_norm_rel, data.Dty_tmp.lpNorm<Eigen::Infinity>());
            
            dual_res_norm_rel = std::max(
                dual_res_norm_rel, data.Dty_tmp.lpNorm<Eigen::Infinity>());
        }
        else {
            data.prim_res.setZero();
            data.dual_res.setZero();
        }

        prim_res_norm = std::max(
            prim_res_norm, 
            data.prim_res.lpNorm<Eigen::Infinity>());
        dual_res_norm = std::max(
            dual_res_norm, 
            data.dual_res.lpNorm<Eigen::Infinity>());
    }

    inline static void residual_terminal_step(const Knotpoint& kpoint, 
                                              CLQRKernelData& data,
                                              scalar& prim_res_norm, scalar& dual_res_norm,
                                              scalar& prim_res_norm_rel, scalar& dual_res_norm_rel) {
        // int nx = kpoint.n;
        if (kpoint.n_con > 0) {
            data.prim_res = data.Dw_tmp - data.z;

            // data.Hw_tmp.noalias()    = kpoint.H * data.w;
            // data.Dty_tmp.noalias()   = kpoint.D_con.transpose() * data.y;

            // data.dual_res            = data.Hw_tmp;
            // data.dual_res.noalias() += kpoint.h;
            // data.dual_res.noalias() += data.Dty_tmp;

            // data.dual_res.tail(nx)  -= data.lambda;

            data.Dty_tmp.noalias()   = kpoint.D_con.transpose() * data.y;
            data.dual_res = kpoint.D_con * data.rho_vec.cwiseProduct(data.z - data.z_prev);

            prim_res_norm_rel = std::max(
                prim_res_norm_rel, data.Dw_tmp.lpNorm<Eigen::Infinity>());
            prim_res_norm_rel = std::max(
                prim_res_norm_rel, data.z.lpNorm<Eigen::Infinity>());

            // dual_res_norm_rel = std::max(
            //     dual_res_norm_rel, data.Hw_tmp.lpNorm<Eigen::Infinity>());
            // dual_res_norm_rel = std::max(
            //     dual_res_norm_rel, kpoint.h.lpNorm<Eigen::Infinity>());
            // dual_res_norm_rel = std::max(
            //     dual_res_norm_rel, data.Dty_tmp.lpNorm<Eigen::Infinity>());

            dual_res_norm_rel = std::max(
                dual_res_norm_rel, data.Dty_tmp.lpNorm<Eigen::Infinity>());
        }
        else {
            data.prim_res.setZero();
            data.dual_res.setZero();
        }

        prim_res_norm = std::max(
            prim_res_norm, data.prim_res.lpNorm<Eigen::Infinity>());
        dual_res_norm = std::max(
            dual_res_norm, data.dual_res.lpNorm<Eigen::Infinity>());
    }

    inline static void update_rho_step(const Knotpoint& kpoint, CLQRKernelData& data, scalar rho_tmp, scalar rho_min) {
        for (int i = 0; i < data.rho_vec.size(); ++i) {
            if (kpoint.e_lb[i] == -LQR_INFTY && kpoint.e_ub[i] == LQR_INFTY) {
                data.rho_vec[i] = rho_min;
                data.inv_rho_vec[i] = 1.0 / rho_min;
            }
            else if (abs(kpoint.e_lb[i] - kpoint.e_ub[i]) < 1e-6) {
                data.rho_vec[i] = 1e3 * rho_tmp;
                data.inv_rho_vec[i] = 1.0 / (1e3 * rho_tmp);
            }
            else if (kpoint.e_lb[i] < kpoint.e_ub[i]) {
                data.rho_vec[i] = rho_tmp;
                data.inv_rho_vec[i] = 1.0 / rho_tmp;
            }
        }
    }
};

} // namespace lqr