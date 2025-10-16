#pragma once

#include "clqr/typedefs.hpp"

namespace lqr {

struct LQRSettings
{
    // termination parameters
    int max_iter   = 1000;
    scalar eps_abs = 1e-5;
    scalar eps_rel = 1e-5;
    int termination_check_interval = 25;

    // ADMM parameters
    scalar regu  = 1e-6;
    scalar sigma = 1e-6;
    scalar alpha = 1.6;

    int rho_update_interval = 50;
    scalar adaptive_rho_tolerance = 5.0;
    scalar rho = 0.1;
    scalar rho_min = 1e-6;
    scalar rho_max = 1e+3;

    // // Load settings from config file
    // static LQRSettings fromConfig(const std::string& config_file_path = "config/default_config.json");
    
    // // Load settings from ConfigManager
    // static LQRSettings fromConfigManager();
};

} // namespace lqr
