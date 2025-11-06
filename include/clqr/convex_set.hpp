// #pragma once

// #include <memory>
// #include <vector>
// #include <cmath>
// #include <cassert>
// #include "clqr/typedefs.hpp"

// namespace lqr {

// /**
//  * @brief Base class for constraint sets
//  * 
//  * Represents a constraint set K where z ∈ K.
//  * Derived classes implement specific constraint types (box, cone, etc.)
//  */
// class ConvexSet {
// public:
//     virtual ~ConvexSet() = default;

//     /**
//      * @brief Get the dimension of the constraint set
//      */
//     virtual int dim() const = 0;

//     /**
//      * @brief Project a vector onto the constraint set
//      * 
//      * Computes the Euclidean projection: proj_K(v) = argmin_{z ∈ K} ||z - v||²
//      * 
//      * @param v Input vector to project
//      */
//     virtual void project(VectorXs& v) const = 0;

//     /**
//      * @brief Check if a vector is in the constraint set
//      * 
//      * @param z Vector to check
//      * @param tol Tolerance for checking feasibility
//      * @return true if z ∈ K within tolerance
//      */
//     // virtual bool is_feasible(const VectorXs& z, scalar tol = 1e-8) const = 0;

//     /**
//      * @brief Get the type name of the constraint set (for debugging/logging)
//      */
//     virtual std::string type_name() const = 0;
// };

// /**
//  * @brief Box constraint set: lb ≤ z ≤ ub
//  * 
//  * Represents element-wise box constraints on a vector.
//  * Each component z[i] must satisfy: lb[i] ≤ z[i] ≤ ub[i]
//  */
// class Box : public ConvexSet {
// public:
//     /**
//      * @brief Construct a box constraint with specified bounds
//      * 
//      * @param lower_bound Lower bound vector
//      * @param upper_bound Upper bound vector
//      */
//     Box(const VectorXs& lower_bound, const VectorXs& upper_bound)
//         : lb_(lower_bound), ub_(upper_bound) {
//         if (lb_.size() != ub_.size()) {
//             throw std::runtime_error("Lower and upper bounds must have the same dimension");
//         }
//         dim_ = lb_.size();
//     }

//     /**
//      * @brief Construct a box constraint with uniform bounds
//      * 
//      * @param dimension Dimension of the constraint
//      * @param lower_bound Scalar lower bound applied to all components
//      * @param upper_bound Scalar upper bound applied to all components
//      */
//     Box(int dimension, scalar lower_bound = -LQR_INFTY, scalar upper_bound = LQR_INFTY)
//         : dim_(dimension) {
//         lb_.resize(dimension);
//         ub_.resize(dimension);
//         lb_.setConstant(lower_bound);
//         ub_.setConstant(upper_bound);
//     }

//     int dim() const override { return dim_; }

//     void project(VectorXs& v) const override {
//         assert(v.size() == dim_ && "Input vector dimension mismatch");
//         v = v.cwiseMax(lb_).cwiseMin(ub_);
//     }

//     // bool is_feasible(const VectorXs& z, scalar tol = 1e-8) const override {
//     //     if (z.size() != dim_) {
//     //         return false;
//     //     }
//     //     for (int i = 0; i < dim_; ++i) {
//     //         if (z[i] < lb_[i] - tol || z[i] > ub_[i] + tol) {
//     //             return false;
//     //         }
//     //     }
//     //     return true;
//     // }

//     std::string type_name() const override { return "BoxConstraint"; }

//     // Accessors
//     const VectorXs& lower_bound() const { return lb_; }
//     const VectorXs& upper_bound() const { return ub_; }

//     // Setters
//     void set_lower_bound(const VectorXs& lb) {
//         if (lb.size() != dim_) {
//             throw std::runtime_error("Lower bound dimension mismatch");
//         }
//         lb_ = lb;
//     }

//     void set_upper_bound(const VectorXs& ub) {
//         if (ub.size() != dim_) {
//             throw std::runtime_error("Upper bound dimension mismatch");
//         }
//         ub_ = ub;
//     }

//     void set_bounds(const VectorXs& lb, const VectorXs& ub) {
//         if (lb.size() != ub.size() || lb.size() != dim_) {
//             throw std::runtime_error("Bounds dimension mismatch");
//         }
//         lb_ = lb;
//         ub_ = ub;
//     }

// private:
//     int dim_;
//     VectorXs lb_;  // Lower bounds
//     VectorXs ub_;  // Upper bounds
// };

// // /**
// //  * @brief Second-Order Cone (Lorentz Cone) constraint: ||z[1:n-1]||₂ ≤ z[0]
// //  * 
// //  * The second-order cone (also called Lorentz cone or ice-cream cone) is defined as:
// //  * K = {z ∈ ℝⁿ : ||z[1:n-1]||₂ ≤ z[0]}
// //  * 
// //  * where z[0] is the scalar part and z[1:n-1] is the vector part.
// //  */
// // class SecondOrderCone : public ConstraintSet {
// // public:
// //     /**
// //      * @brief Construct a second-order cone constraint
// //      * 
// //      * @param dimension Dimension of the cone (must be at least 2)
// //      */
// //     explicit SecondOrderCone(int dimension) : dim_(dimension) {
// //         if (dimension < 2) {
// //             throw std::runtime_error("Second-order cone dimension must be at least 2");
// //         }
// //     }

// //     int dim() const override { return dim_; }

// //     void project(const VectorXs& v, VectorXs& result) const override {
// //         if (v.size() != dim_) {
// //             throw std::runtime_error("Input vector dimension mismatch");
// //         }
// //         result.resize(dim_);

// //         const scalar t = v[0];  // Scalar part
// //         const auto x = v.segment(1, dim_ - 1);  // Vector part
// //         const scalar x_norm = x.norm();

// //         // Projection onto the second-order cone
// //         if (x_norm <= t) {
// //             // v is already in the cone
// //             result = v;
// //         } else if (x_norm <= -t) {
// //             // v is in the polar cone, project to origin
// //             result.setZero();
// //         } else {
// //             // General case: project onto the cone boundary
// //             const scalar alpha = 0.5 * (1.0 + t / x_norm);
// //             result[0] = alpha * x_norm;
// //             result.segment(1, dim_ - 1) = alpha * x;
// //         }
// //     }

// //     bool is_feasible(const VectorXs& z, scalar tol = 1e-8) const override {
// //         if (z.size() != dim_) {
// //             return false;
// //         }
// //         const scalar t = z[0];
// //         const auto x = z.segment(1, dim_ - 1);
// //         const scalar x_norm = x.norm();
// //         return x_norm <= t + tol;
// //     }

// //     std::string type_name() const override { return "SecondOrderCone"; }

// // private:
// //     int dim_;
// // };

// /**
//  * @brief Product constraint set: Cartesian product of multiple constraint sets
//  * 
//  * Represents K = K₁ × K₂ × ... × Kₙ where each Kᵢ is a constraint set.
//  * Useful for combining different types of constraints.
//  */
// class ProductConstraint : public ConstraintSet {
// public:
//     ProductConstraint() : dim_(0) {}

//     /**
//      * @brief Add a constraint set to the product
//      */
//     void add_constraint(std::shared_ptr<ConstraintSet> constraint) {
//         constraints_.push_back(constraint);
//         offsets_.push_back(dim_);
//         dim_ += constraint->dim();
//     }

//     int dim() const override { return dim_; }

//     void project(VectorXs& v) const override {
//        assert(v.size() == dim_ && "Input vector dimension mismatch");
//         for (size_t i = 0; i < constraints_.size(); ++i) {
//             const int offset = offsets_[i];
//             const int constraint_dim = constraints_[i]->dim();
//             VectorRef v_i = v.segment(offset, constraint_dim);
//             constraints_[i]->project(v_i);
//         }
//     }

//     // bool is_feasible(const VectorXs& z, scalar tol = 1e-8) const override {
//     //     if (z.size() != dim_) {
//     //         return false;
//     //     }

//     //     for (size_t i = 0; i < constraints_.size(); ++i) {
//     //         const int offset = offsets_[i];
//     //         const int constraint_dim = constraints_[i]->dim();
//     //         VectorXs z_segment = z.segment(offset, constraint_dim);
            
//     //         if (!constraints_[i]->is_feasible(z_segment, tol)) {
//     //             return false;
//     //         }
//     //     }
//     //     return true;
//     // }

//     std::string type_name() const override { return "ProductConstraint"; }

//     // Accessor
//     const std::vector<std::shared_ptr<ConstraintSet>>& get_constraints() const {
//         return constraints_;
//     }

//     size_t num_constraints() const { return constraints_.size(); }

// private:
//     std::vector<std::shared_ptr<ConstraintSet>> constraints_;
//     std::vector<int> offsets_;  // Starting index for each constraint
//     int dim_;
// };

// // /**
// //  * @brief Free constraint set: ℝⁿ (no constraints)
// //  * 
// //  * Represents the trivial constraint where any vector is feasible.
// //  * Projection is the identity operation.
// //  */
// // class FreeConstraint : public ConstraintSet {
// // public:
// //     explicit FreeConstraint(int dimension) : dim_(dimension) {}

// //     int dim() const override { return dim_; }

// //     void project(const VectorXs& v, VectorXs& result) const override {
// //         if (v.size() != dim_) {
// //             throw std::runtime_error("Input vector dimension mismatch");
// //         }
// //         result = v;  // Identity projection
// //     }

// //     bool is_feasible(const VectorXs& z, scalar tol = 1e-8) const override {
// //         return z.size() == dim_;  // Any vector of correct dimension is feasible
// //     }

// //     std::string type_name() const override { return "FreeConstraint"; }

// // private:
// //     int dim_;
// // };

// // /**
// //  * @brief Zero constraint set: {0} (equality to zero)
// //  * 
// //  * Represents the constraint z = 0.
// //  * Projection always returns the zero vector.
// //  */
// // class ZeroConstraint : public ConstraintSet {
// // public:
// //     explicit ZeroConstraint(int dimension) : dim_(dimension) {}

// //     int dim() const override { return dim_; }

// //     void project(const VectorXs& v, VectorXs& result) const override {
// //         if (v.size() != dim_) {
// //             throw std::runtime_error("Input vector dimension mismatch");
// //         }
// //         result.resize(dim_);
// //         result.setZero();
// //     }

// //     bool is_feasible(const VectorXs& z, scalar tol = 1e-8) const override {
// //         if (z.size() != dim_) {
// //             return false;
// //         }
// //         return z.norm() <= tol;
// //     }

// //     std::string type_name() const override { return "ZeroConstraint"; }

// // private:
// //     int dim_;
// // };

// } // namespace lqr
