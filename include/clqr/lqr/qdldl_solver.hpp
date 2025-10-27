#pragma once

#include <qdldl/qdldl.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include "clqr/lqr_model.hpp"
#include "qdldl_typedefs.hpp"
#include "kkt.hpp"

namespace lqr
{

class QDLDLSolver
{
public:
    QDLDLSolver(const LQRModel& model);
    std::unique_ptr<QDLDLData> create_workspace(const CscMatrix& Kkt);
    void update_problem_data(const std::vector<VectorXs>& ws, 
                             const std::vector<VectorXs>& ys, 
                             const std::vector<VectorXs>& zs,
                             const std::vector<VectorXs>& inv_rho_vecs,
                             const scalar sigma);
    void backward(const std::vector<VectorXs>& inv_rho_vecs);
    // void backward_without_factorization(const std::vector<VectorXs>& rho_vecs);
    void forward(const VectorXs& x0, std::vector<VectorXs>& ws);

private:
    const LQRModel& model_;
    std::unique_ptr<KKTSystem> KKT_system_;
    std::unique_ptr<CscMatrix> KKT_csc_;
    std::unique_ptr<QDLDLData> qdldl_data_;

};

/* Definitions */
QDLDLSolver::QDLDLSolver(const LQRModel& model) 
    : model_(model)
{
    KKT_system_ = std::make_unique<KKTSystem>(model.n, model.m, model.N, model.ncs);
    scalar rho_dyn = 1e-6;
    scalar sigma   = 1e-6;
    KKT_system_->form_KKT_system(model_, rho_dyn, sigma, false);
    KKT_csc_ = KKT_system_->get_KKT_csc_matrix();
    qdldl_data_ = create_workspace(*KKT_csc_);
}

std::unique_ptr<QDLDLData> QDLDLSolver::create_workspace(const CscMatrix& K)
{
    auto data = std::make_unique<QDLDLData>();
    
    QDLDL_int n = K.n; // Number of columns 
    data->Ln = n;

    // Allocate arrays using smart pointers    
    data->etree = std::make_unique<QDLDL_int[]>(n);
    data->Lnz   = std::make_unique<QDLDL_int[]>(n);
    
    data->Lp    = std::make_unique<QDLDL_int[]>(n + 1);
    data->D     = std::make_unique<QDLDL_float[]>(n);
    data->Dinv  = std::make_unique<QDLDL_float[]>(n);
    
    data->iwork = std::make_unique<QDLDL_int[]>(3 * n);
    data->bwork = std::make_unique<QDLDL_bool[]>(n);
    data->fwork = std::make_unique<QDLDL_float[]>(n);
    
    data->sumLnz = QDLDL_etree(n, K.p, K.i, data->iwork.get(), data->Lnz.get(), data->etree.get());
            
    if (data->sumLnz < 0) {
        throw std::runtime_error("Error in QDLDL_etree");
    }

    data->Li = std::make_unique<QDLDL_int[]>(data->sumLnz);
    data->Lx = std::make_unique<QDLDL_float[]>(data->sumLnz);
    
    data->x  = std::make_unique<QDLDL_float[]>(n);
        
    return data;
}

void QDLDLSolver::update_problem_data(const std::vector<VectorXs>& ws, 
                                      const std::vector<VectorXs>& ys, 
                                      const std::vector<VectorXs>& zs,
                                      const std::vector<VectorXs>& inv_rho_vecs, 
                                      const scalar sigma) {
    KKT_system_->form_rhs(model_, ws, ys, zs, inv_rho_vecs, sigma);
}

void QDLDLSolver::backward(const std::vector<VectorXs>& inv_rho_vecs) {
    KKT_system_->update_rho_vecs(model_, inv_rho_vecs);
    QDLDL_int fact_status = QDLDL_factor(
        KKT_csc_.get()->n,
        KKT_csc_.get()->p,
        KKT_csc_.get()->i,
        KKT_csc_.get()->x,
        qdldl_data_.get()->Lp.get(),
        qdldl_data_.get()->Li.get(),
        qdldl_data_.get()->Lx.get(),
        qdldl_data_.get()->D.get(),
        qdldl_data_.get()->Dinv.get(),
        qdldl_data_.get()->Lnz.get(),
        qdldl_data_.get()->etree.get(),
        qdldl_data_.get()->bwork.get(),
        qdldl_data_.get()->iwork.get(),
        qdldl_data_.get()->fwork.get()
    );
    if (fact_status < 0) {
        throw std::runtime_error("QDLDL factorization failed with status: " + std::to_string(fact_status));
    }
}

void QDLDLSolver::forward(const VectorXs& x0, std::vector<VectorXs>& ws) {
    /*
    ** ws = [u0; x0; u1; x1; ...; uN-1; xN]
    */
    KKT_system_->update_initial_stage_rhs(model_, x0);
    // Use Eigen::Map to avoid element-by-element copy (both are double)
    Eigen::Map<VectorXs>(qdldl_data_->x.get(), KKT_csc_->n) = KKT_system_->get_rhs();
    QDLDL_solve(
        qdldl_data_->Ln,
        qdldl_data_->Lp.get(),
        qdldl_data_->Li.get(),
        qdldl_data_->Lx.get(),
        qdldl_data_->Dinv.get(),
        qdldl_data_->x.get()
    );
    // Copy solution back to ws
    const int nx = model_.n;
    const int nu = model_.m;
    const int N  = model_.N;
    /*
    ** x = [
        u0, 
        x1, u1,
        ...
        xN,
        y0,
        lambda1, y1,
        ...
        lambdaN, yN
    ]
    */
    ws[0].tail(nx) = x0;
    ws[0].head(nu) = Eigen::Map<const VectorXs>(qdldl_data_->x.get(), nu);
    int offset = nu;
    for (int k = 1; k < N; ++k) {
        // x
        ws[k].tail(nx) = Eigen::Map<VectorXs>(qdldl_data_->x.get() + offset, nx);
        // u
        ws[k].head(nu) = Eigen::Map<VectorXs>(qdldl_data_->x.get() + offset + nx, nu);
        // 
        offset += nx + nu;
    }
    // Terminal state
    ws[N].tail(nx) = Eigen::Map<VectorXs>(qdldl_data_->x.get() + offset, nx);
}

} // namespace lqr