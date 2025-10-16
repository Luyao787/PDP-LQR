#pragma once 

#include "clqr/typedefs.hpp"
#include "clqr/lqr_model.hpp"

namespace lqr {

struct OSCLQRKernelData 
{
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    OSCLQRKernelData(int nx, int nu, int nc, bool is_terminal_stage=false):
        is_terminal(is_terminal_stage) 
    {
        int nxu = nx + nu;
        if (nc > 0) {
           if (is_terminal) {
                Dty_tmp.resize(nx);
                dual_res.resize(nx);
            }
            else {
                Dty_tmp.resize(nxu);
                dual_res.resize(nxu);
            }
            Dw_tmp.resize(nc);
            z_prev.resize(nc);
            z_tilde.resize(nc);
            prim_res.resize(nc);
            y_tmp.resize(nc);
        }
        set_zero();
    }

    void set_zero() {
        if (z_tilde.size() > 0) {
            Dw_tmp.setZero();
            Dty_tmp.setZero();
            prim_res.setZero();
            dual_res.setZero();
            z_prev.setZero();
            z_tilde.setZero();
            y_tmp.setZero();
        }
    }

    bool is_terminal;

    VectorXs Dw_tmp; 
    VectorXs Dty_tmp;
    VectorXs prim_res, dual_res;
    VectorXs y_tmp;

    VectorXs z_prev, z_tilde;

};

struct OSCLQRKernel
{
    inline static void residual_step(const Node& kpoint,
                                     const VectorXs& y, const VectorXs& z, const VectorXs& rho_vec,
                                     OSCLQRKernelData& data,
                                     scalar& prim_res_norm, scalar& dual_res_norm,
                                     scalar& prim_res_norm_rel, scalar& dual_res_norm_rel) {
        if (kpoint.n_con > 0) {
            data.prim_res = data.Dw_tmp - z;
      
            data.Dty_tmp.noalias() = kpoint.D_con.transpose() * y;
            data.dual_res = kpoint.D_con * rho_vec.cwiseProduct(z - data.z_prev);

            prim_res_norm_rel = std::max(
                prim_res_norm_rel, data.Dw_tmp.lpNorm<Eigen::Infinity>());
            prim_res_norm_rel = std::max(
                prim_res_norm_rel, z.lpNorm<Eigen::Infinity>());
             
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

    inline static void update_rho_step(const Node& kpoint, 
                                       VectorXs& rho_vec, VectorXs& inv_rho_vec, 
                                       scalar rho_tmp, scalar rho_min) {
        for (int i = 0; i < rho_vec.size(); ++i) {
            if (kpoint.e_lb[i] == -LQR_INFTY && kpoint.e_ub[i] == LQR_INFTY) {
                rho_vec[i] = rho_min;
                inv_rho_vec[i] = 1.0 / rho_min;
            }
            else if (abs(kpoint.e_lb[i] - kpoint.e_ub[i]) < 1e-6) {
                rho_vec[i] = 1e3 * rho_tmp;
                inv_rho_vec[i] = 1.0 / (1e3 * rho_tmp);
            }
            else if (kpoint.e_lb[i] < kpoint.e_ub[i]) {
                rho_vec[i] = rho_tmp;
                inv_rho_vec[i] = 1.0 / rho_tmp;
            }
        }
    }
};

} // namespace lqr