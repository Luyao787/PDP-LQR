#pragma once

#include "lqr/lqr_model.hpp"
#include "lqr/typedefs.hpp"
#include "lqr/clqr_kernel.hpp"

namespace lqr {

struct ParallelCLQRKernelData : public CLQRKernelData
{
    MatrixXs G, F, C;
    VectorXs f;

    MatrixXs BtFt, F_tmp;
    VectorXs f_tmp;

    MatrixXs K;
    VectorXs d;

    ParallelCLQRKernelData(int nx, int nu, int nc, bool is_terminal_stage=false)
        : CLQRKernelData(nx, nu, nc, is_terminal_stage)
    {
        G.resize(nu, nx);
        F.resize(nx, nx);
        C.resize(nx, nx);
        f.resize(nx);
        BtFt.resize(nu, nx);
        F_tmp.resize(nx, nx);
        f_tmp.resize(nx);
        K.resize(nu, nx);
        d.resize(nu);
    }
};

struct ParallelCLQRCondensedKernelData
{
    MatrixXs A, C, Q, P;
    VectorXs c, q, p;

    MatrixXs PC, PA, D;
    Eigen::PartialPivLU<MatrixXs> lu_fact;
    VectorXs Pc, e;

    VectorXs xhat, uhat;

    ParallelCLQRCondensedKernelData(int nx)
    {
        A.resize(nx, nx);
        C.resize(nx, nx);
        Q.resize(nx, nx);
        P.resize(nx, nx);
        c.resize(nx);
        q.resize(nx);
        p.resize(nx);
        PC.resize(nx, nx);
        PA.resize(nx, nx);
        D.resize(nx, nx);
        Pc.resize(nx);
        e.resize(nx);
        xhat.resize(nx);
        uhat.resize(nx);
    } 
};

struct ParallelCLQRKernel {

    inline static 
    void terminal_step_with_factorization(const Knotpoint& kpoint,
                                          ParallelCLQRKernelData& data, 
                                          scalar sigma, 
                                          bool is_last_segment) {
        if (is_last_segment) {
           CLQRKernel::terminal_step_with_factorization(kpoint, 
                                                        static_cast<CLQRKernelData&>(data), 
                                                        sigma);
        } else {
            data.L.setZero();
            data.lp.setZero();
            data.C.setZero();
            data.f.setZero();
            data.F.setIdentity();
        }
    }

    inline static void terminal_step_without_factorization(const Knotpoint& kpoint, ParallelCLQRKernelData& data, scalar sigma, bool is_last_segment) {
        if (is_last_segment) {
            CLQRKernel::terminal_step_without_factorization(kpoint, 
                                                            static_cast<CLQRKernelData&>(data), 
                                                            sigma);
        } else {
            data.L.setZero();
            data.lp.setZero();
            data.C.setZero();
            data.f.setZero();
            data.F.setIdentity();
        }
    }

    inline static void step_with_factorization(const Knotpoint& kpoint, ParallelCLQRKernelData& data, const ParallelCLQRKernelData& data_next, scalar sigma, bool is_last_segment) {
        
        CLQRKernel::step_with_factorization(kpoint, 
                                            static_cast<CLQRKernelData&>(data), 
                                            static_cast<const CLQRKernelData&>(data_next), 
                                            sigma);
        if (!is_last_segment) {
            int nx = kpoint.n;
            int nu = kpoint.m;
            MatrixRef K   = data.K;
            VectorRef d   = data.d;
            MatrixRef Lxu = data.L.bottomLeftCorner(nx, nu);
            MatrixRef Luu = data.L.topLeftCorner(nu, nu);
            VectorRef lu  = data.lp.head(nu);            
            K = -Lxu.transpose();
            d = -lu;
            Luu.triangularView<Eigen::Lower>().transpose().solveInPlace(K); // nx nu^2
            Luu.triangularView<Eigen::Lower>().transpose().solveInPlace(d);

            ConstMatrixRef F_next = data_next.F;
            ConstMatrixRef C_next = data_next.C;
            ConstVectorRef f_next = data_next.f;
            ConstMatrixRef A      = kpoint.E.topRightCorner(nx, nx);
            ConstMatrixRef B      = kpoint.E.topLeftCorner(nx, nu);
            ConstVectorRef c      = kpoint.c;

            MatrixRef G = data.G;
            MatrixRef F = data.F;
            MatrixRef C = data.C;
            VectorRef f = data.f;

            MatrixRef BtFt  = data.BtFt;
            MatrixRef F_tmp = data.F_tmp;
            VectorRef f_tmp = data.f_tmp;

            BtFt.noalias()  = B.transpose() * F_next.transpose(); // 2 nx^2 nu 
            G               = -BtFt;
            Luu.triangularView<Eigen::Lower>().solveInPlace(G); // nx nu^2
            F_tmp.noalias() = A + B * K; // 2 nx^2 nu
            F.noalias()     = F_next * F_tmp; // 2 nx^3
            f_tmp.noalias() = c + B * d; 
            f.noalias()     = F_next * f_tmp + f_next;
            C.noalias()     = C_next + G.transpose() * G; // 2 nx^2 nu
        }
    }

    inline static void step_without_factorization(const Knotpoint& kpoint, ParallelCLQRKernelData& data, const ParallelCLQRKernelData& data_next, scalar sigma, bool is_last_segment) {
      
        CLQRKernel::step_without_factorization(kpoint, 
                                               static_cast<CLQRKernelData&>(data), 
                                               static_cast<const CLQRKernelData&>(data_next), 
                                               sigma);
        if (!is_last_segment) {
            int nx = kpoint.n;
            int nu = kpoint.m;
            VectorRef d   = data.d;
            MatrixRef Luu = data.L.topLeftCorner(nu, nu);
            VectorRef lu  = data.lp.head(nu);            
            d = -lu;
            Luu.triangularView<Eigen::Lower>().transpose().solveInPlace(d);

            ConstMatrixRef F_next = data_next.F;
            ConstVectorRef f_next = data_next.f;
            ConstMatrixRef B      = kpoint.E.topLeftCorner(nx, nu);
            ConstVectorRef c      = kpoint.c;

            VectorRef f     = data.f;
            VectorRef f_tmp = data.f_tmp;

            f_tmp.noalias() = c + B * d;
            f.noalias()     = F_next * f_tmp + f_next;
        }
    }

    inline static void forward_step(const Knotpoint& kpoint, 
                                    ParallelCLQRKernelData& data, 
                                    ParallelCLQRKernelData& data_next,
                                    ParallelCLQRCondensedKernelData& condensed_data,
                                    bool is_last_segment) 
    {
        if (is_last_segment) {
            CLQRKernel::forward_step(kpoint, 
                                     static_cast<CLQRKernelData&>(data), 
                                     static_cast<CLQRKernelData&>(data_next));
        } else {
            int nx = kpoint.n;
            int nu = kpoint.m;
            ConstMatrixRef G_k = data.G;
            ConstMatrixRef A_k = kpoint.E.topRightCorner(nx, nx);
            ConstMatrixRef B_k = kpoint.E.topLeftCorner(nx, nu);
            ConstVectorRef c_k = kpoint.c;

            MatrixRef      Luu_k = data.L.topLeftCorner(nu, nu);
            ConstMatrixRef Lxu_k = data.L.bottomLeftCorner(nx, nu);
            ConstVectorRef lu_k  = data.lp.head(nu);
            
            VectorRef x      = data.w.tail(nx);
            VectorRef u      = data.w.head(nu);
            VectorRef x_next = data_next.w.tail(nx);
            VectorRef uhat   = condensed_data.uhat;

            u.noalias()  = -lu_k;
            u.noalias() -= Lxu_k.transpose() * x;
            u.noalias() += G_k * uhat;
            Luu_k.triangularView<Eigen::Lower>().transpose().solveInPlace(u);

            x_next.noalias()  = c_k;
            x_next.noalias() += A_k * x;
            x_next.noalias() += B_k * u;

            // Dual variable update
            ConstMatrixRef F_next = data_next.F;
            ConstMatrixRef Lxx_next = data_next.L.bottomRightCorner(nx, nx);
            ConstVectorRef p_next   = data_next.lp.tail(nx);
            VectorRef lambda_next   = data_next.lambda;

            lambda_next.noalias()  = Lxx_next.transpose() * x_next;
            lambda_next            = Lxx_next * lambda_next; // more efficient?
            lambda_next           += p_next;
            lambda_next.noalias() += F_next.transpose() * uhat;
        }
    }
};

} // namespace lqr