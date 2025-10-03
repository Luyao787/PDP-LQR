#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "tests/test_utils.hpp"

using namespace lqr;

class ConstrainedLQRTest : public ::testing::Test {
protected:
    void SetUp() override {
        model = create_LQR_model();
        nx = model->n;
        nu = model->m;
        N = model->N;
        ncs_.resize(N + 1, 0);
    }
    
    void LoadTrajectories(std::vector<VectorXs>& sol_xs, std::vector<VectorXs>& sol_us) {
        using json = nlohmann::json;
        
        // Try multiple possible paths for the data files
        std::vector<std::string> x_paths = {
            "../tests/data/lqr_quadrotor_x_trj.json",     // from build/ directory
            "tests/data/lqr_quadrotor_x_trj.json",        // from project root
            "../../tests/data/lqr_quadrotor_x_trj.json",  // from build/tests/ directory (ctest)
        };
        
        std::ifstream x_trj_file;
        for (const auto& path : x_paths) {
            x_trj_file.open(path);
            if (x_trj_file.is_open()) break;
        }
        ASSERT_TRUE(x_trj_file.is_open()) << "Could not open state trajectory JSON file. Tried relative paths: ../tests/data/, tests/data/, and ../../tests/data/";
        
        json x_trj_json;
        x_trj_file >> x_trj_json;
        x_trj_file.close();
        
        sol_xs.clear();
        ASSERT_TRUE(x_trj_json.is_array()) << "State trajectory JSON must be an array";
        
        for (const auto& state_array : x_trj_json) {
            ASSERT_TRUE(state_array.is_array()) << "Each state must be an array";
            ASSERT_EQ(state_array.size(), nx) << "State dimension mismatch";
            
            VectorXs x(nx);
            for (int i = 0; i < nx; ++i) {
                x(i) = state_array[i].get<double>();
            }
            sol_xs.push_back(x);
        }
        
        // Load control trajectory from JSON
        std::vector<std::string> u_paths = {
            "../tests/data/lqr_quadrotor_u_trj.json",     // from build/ directory
            "tests/data/lqr_quadrotor_u_trj.json",        // from project root
            "../../tests/data/lqr_quadrotor_u_trj.json",  // from build/tests/ directory (ctest)
        };
        
        std::ifstream u_trj_file;
        for (const auto& path : u_paths) {
            u_trj_file.open(path);
            if (u_trj_file.is_open()) break;
        }
        ASSERT_TRUE(u_trj_file.is_open()) << "Could not open control trajectory JSON file. Tried relative paths: ../tests/data/, tests/data/, and ../../tests/data/";
        
        json u_trj_json;
        u_trj_file >> u_trj_json;
        u_trj_file.close();
        
        sol_us.clear();
        ASSERT_TRUE(u_trj_json.is_array()) << "Control trajectory JSON must be an array";
        
        for (const auto& control_array : u_trj_json) {
            ASSERT_TRUE(control_array.is_array()) << "Each control must be an array";
            ASSERT_EQ(control_array.size(), nu) << "Control dimension mismatch";
            
            VectorXs u(nu);
            for (int i = 0; i < nu; ++i) {
                u(i) = control_array[i].get<double>();
            }
            sol_us.push_back(u);
        }
        
        ASSERT_EQ(sol_xs.size(), N + 1) << "State trajectory has wrong length";
        ASSERT_EQ(sol_us.size(), N) << "Control trajectory has wrong length";
    }
    
    int nx, nu, N;
    std::vector<int> ncs_;
    std::unique_ptr<LQRModel> model;
};

TEST_F(ConstrainedLQRTest, ConstrainedLQRSolve) {
    SerialCLQRSolver solver(*model);
    ParallelCLQRSolver par_solver(*model, 2); // using 2 segments for parallel solver

    LQRSettings settings;
    settings.max_iter = 1000;
    settings.eps_abs = 1e-5;
    settings.eps_rel = 1e-5;
    
    LQRResults results, results_par;
    results.reset(nx, nu, ncs_, N);
    results_par.reset(nx, nu, ncs_, N);
    
    VectorXs x0(nx);
    x0 << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
    
    EXPECT_NO_THROW(solver.solve(x0, settings, results));
    EXPECT_NO_THROW(par_solver.solve(x0, settings, results_par));
    
    // Load trajectory data for comparison
    std::vector<VectorXs> sol_xs, sol_us;
    LoadTrajectories(sol_xs, sol_us);
    
    // Compare solutions
    ASSERT_EQ(sol_xs.size(), N + 1);
    ASSERT_EQ(sol_us.size(), N);
    
    for (int k = 0; k <= N; ++k) {
        for (int i = 0; i < nx; ++i) {
            EXPECT_NEAR(results.xs[k](i), sol_xs[k](i), 1e-4);
            EXPECT_NEAR(results_par.xs[k](i), sol_xs[k](i), 1e-4);
        }
    }
    
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < nu; ++i) {
            EXPECT_NEAR(results.us[k](i), sol_us[k](i), 1e-4);
            EXPECT_NEAR(results_par.us[k](i), sol_us[k](i), 1e-4);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}