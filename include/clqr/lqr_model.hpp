#pragma once

#include <vector>
#include "clqr/typedefs.hpp"

namespace lqr {

struct Node {
    int n, m;  // n: state dimension, m: control dimension
    int n_con; // number of constraints

    /* Linear dynamics
        x_{k+1} = A * x_k + B * u_k + c_k */
    MatrixXs E; // E = [B A]
    VectorXs c;

    /* Cost */
    MatrixXs H; // H = [R S; S^T Q]
    VectorXs h; // h = [r; q]

    /* Constraints 
        e_lb <= Du * u + Dx * x <= e_ub */
    MatrixXs D_con; // D_con = [Du Dx]
    VectorXs e_lb, e_ub;

    bool is_terminal;
    int time_step;

    Node(int state_dim, int control_dim, int n_constraints, int time_step, bool is_terminal_stage = false)
        : n(state_dim), m(control_dim), n_con(n_constraints), is_terminal(is_terminal_stage), time_step(time_step)
    {
        if (is_terminal) {
            H.resize(n, n);
            h.resize(n);
        } else {
            E.resize(n, n + m);
            c.resize(n);
            H.resize(n + m, n + m);
            h.resize(n + m);
        }
        if (n_con > 0) {
            D_con.resize(n_con, is_terminal ? n : n + m);
            e_lb.resize(n_con);
            e_ub.resize(n_con);
        }
        set_zero();
    }

    void set_zero() {
        if (!is_terminal) {
            E.setZero();
            c.setZero();
        }
        H.setZero();
        h.setZero();
        if (n_con > 0) {
            D_con.setZero();
            e_lb.setZero();
            e_ub.setZero();
        }
    }

    int get_constraint_dim() const { return n_con; }
};

struct LQRModel {
    int n; // state dimension
    int m; // control dimension
    int N; // number of intervals

    std::vector<int> ncs;
    std::vector<Node> nodes;

    LQRModel(int n, int m, int horizon) : n(n), m(m), N(horizon) {
        if (N < 1) {
            throw std::runtime_error("Horizon must be at least 1.");
        }
        ncs.resize(N + 1);
        nodes.reserve(N + 1);
    }
    
    Node& get_node(int k) { return nodes[k]; }
    const Node& get_node(int k) const { return nodes[k]; }

    void add_node(int n, int m, int nc, int time_step, bool is_terminal_stage = false) {
        nodes.emplace_back(n, m, nc, time_step, is_terminal_stage);
        ncs[time_step] = nc;
    }
};

} // namespace lqr