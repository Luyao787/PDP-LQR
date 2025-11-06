#pragma once

#include <qdldl/qdldl.h>
#include <memory>

namespace lqr {

struct CscMatrix {
    QDLDL_int m;                     // number of rows
    QDLDL_int n;                     // number of cols
    const QDLDL_int* p;              // column pointers (read-only)
    const QDLDL_int* i;              // row indices (read-only)
    const QDLDL_float* x;            // nonzero values (read-only)
    QDLDL_int nzmax;                 // number of nonzeros
};

struct QDLDLData
{
    // data for L and D factors
    QDLDL_int Ln;
    std::unique_ptr<QDLDL_int[]>   Lp;
    std::unique_ptr<QDLDL_int[]>   Li;
    std::unique_ptr<QDLDL_float[]> Lx;
    std::unique_ptr<QDLDL_float[]> D;
    std::unique_ptr<QDLDL_float[]> Dinv;

    // data for elim tree calculation
    std::unique_ptr<QDLDL_int[]> etree;
    std::unique_ptr<QDLDL_int[]> Lnz;
    QDLDL_int sumLnz;

    // working data for factorisation
    std::unique_ptr<QDLDL_int[]>   iwork;
    std::unique_ptr<QDLDL_bool[]>  bwork;
    std::unique_ptr<QDLDL_float[]> fwork;

    // Data for results of A\b
    std::unique_ptr<QDLDL_float[]> x;
};

} // namespace lqr