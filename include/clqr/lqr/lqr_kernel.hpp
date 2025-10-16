#pragma once

#include "clqr/typedefs.hpp"
#include "clqr/lqr_model.hpp"

namespace lqr {

struct LQRKernelData {
    MatrixXs H; 
    VectorXs h;
    VectorXs g;

    /* Variables used during backward pass */
    MatrixXs L;    // L = [Luu Lux; Lxu Lxx]
    MatrixXs V, M; // V = E^T * Lxx, M = H + V * V^T

    VectorXs lp;  
    VectorXs Pb, Pb_tmp;
    
    MatrixXs rhoD;
    VectorXs rhog;

    bool is_terminal;

    LQRKernelData(int nx, int nu, int nc, bool is_terminal_stage=false): 
        is_terminal(is_terminal_stage) {
        int nxu = nx + nu;
        if (is_terminal) {
            H.resize(nx, nx);
            h.resize(nx);
            L.resize(nx, nx);
            lp.resize(nx);
        }
        else {
            H.resize(nxu, nxu);
            h.resize(nxu);
            L.resize(nxu, nxu);
            lp.resize(nxu);
        }

        V.resize(nxu, nx);
        M.resize(nxu, nxu);
        Pb.resize(nx);
        Pb_tmp.resize(nx);

        if (nc > 0) {
            if (is_terminal) { 
                rhoD.resize(nc, nx); 
            }
            else {
                rhoD.resize(nc, nxu);                
            }
            g.resize(nc);
            rhog.resize(nc);
        }
        set_zero();
    }

    void set_zero() {
        H.setZero();
        h.setZero();
        L.setZero();
        lp.setZero();
        V.setZero();
        M.setZero();
        Pb.setZero();
        Pb_tmp.setZero();

        if (g.size() > 0) {
            g.setZero();
            rhoD.setZero();
            rhog.setZero();
        }
    }
};

struct LQRKernel {
    
    inline static 
    void terminal_step_with_factorization(const Node& kpoint, const VectorXs& rho_vec, 
                                          LQRKernelData& data) {
        if (kpoint.n_con > 0) {
            data.rhoD.noalias() = rho_vec.asDiagonal() * kpoint.D_con;
            data.H.noalias() += kpoint.D_con.transpose() * data.rhoD;

            data.rhog = rho_vec.cwiseProduct(data.g);
            data.h.noalias() -= kpoint.D_con.transpose() * data.rhog;
        }
        data.L  = data.H.llt().matrixL();
        data.lp = data.h;
    }

    inline static 
    void terminal_step_without_factorization(const Node& kpoint, const VectorXs& rho_vec, 
                                             LQRKernelData& data) {
        if (kpoint.n_con > 0) {
            data.rhog = rho_vec.cwiseProduct(data.g);
            data.h.noalias() -= kpoint.D_con.transpose() * data.rhog;
        }
        data.lp = data.h;
    }

    inline static 
    void step_with_factorization(const Node& kpoint, const VectorXs& rho_vec, const LQRKernelData& data_next,
                                 LQRKernelData& data) {       
        if (kpoint.n_con > 0) {
            data.rhoD.noalias() = rho_vec.asDiagonal() * kpoint.D_con;
            data.H.noalias() += kpoint.D_con.transpose() * data.rhoD;

            data.rhog = rho_vec.cwiseProduct(data.g);
            data.h.noalias() -= kpoint.D_con.transpose() * data.rhog;
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

    inline static 
    void step_without_factorization(const Node& kpoint, const VectorXs& rho_vec, const LQRKernelData& data_next, 
                                    LQRKernelData& data) { 
        if (kpoint.n_con > 0) {
            data.rhog = rho_vec.cwiseProduct(data.g);
            data.h.noalias() -= kpoint.D_con.transpose() * data.rhog;
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

    inline static 
    void forward_step(const Node& kpoint, 
                      LQRKernelData& data, VectorXs& w, VectorXs& w_next) {
        int nx = kpoint.n;
        int nu = kpoint.m;
        
        ConstMatrixRef A   = kpoint.E.rightCols(nx); 
        ConstMatrixRef B   = kpoint.E.leftCols(nu);   
        ConstVectorRef c   = kpoint.c;
        MatrixRef      Luu = data.L.topLeftCorner(nu, nu);
        ConstMatrixRef Lxu = data.L.bottomLeftCorner(nx, nu);
        ConstVectorRef lu  = data.lp.head(nu);

        VectorRef x      = w.tail(nx);
        VectorRef u      = w.head(nu);
        VectorRef x_next = w_next.tail(nx);
        
        u = -lu;
        u.noalias() -= Lxu.transpose() * x;
        Luu.triangularView<Eigen::Lower>().transpose().solveInPlace(u);

        x_next            = c;
        x_next.noalias() += A * x;
        x_next.noalias() += B * u;

        // ConstMatrixRef Lxx_next = data_next.L.bottomRightCorner(nx, nx);
        // ConstVectorRef p_next   = data_next.lp.tail(nx);
        // VectorRef lambda_next   = data_next.lambda;

        // lambda_next.noalias() = Lxx_next.transpose() * x_next;
        // lambda_next           = Lxx_next * lambda_next; // more efficient?
        // lambda_next          += p_next;
    }
};

} // namespace lqr