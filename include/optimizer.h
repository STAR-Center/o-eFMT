#ifndef IMREG_FMT_OPTIMIZER_H
#define IMREG_FMT_OPTIMIZER_H

#include "ceres/ceres.h"
#include "utility.h"
#include <iostream>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

class Optimizer {
  protected:
    Eigen::Matrix3d K;
    // g2o::SparseOptimizer optimizer;
    int curr_frame = 0;
    int windows_sz = 5;
    double sum_x = 0;
    double sum_y = 0;
    double sum_lambda = 0;
    double st_x = 0;
    double st_y = 0;
    std::queue<RegisTf> windows;
    RegisTf mean;

  public:
    explicit Optimizer(const double& fx, const double& fy){sum_x = 0; sum_y=0; st_x=0; st_y=0; sum_lambda = 0;}
    ~Optimizer(){};
    //    void optimize(Eigen::Matrix4d &pose, Eigen::Matrix4d &opt_pose2, Eigen::Matrix4d &tf01, Eigen::Matrix4d &tf02,
    //                  Eigen::Matrix4d &tf12);
    // void check_before_optimize(RegisTf& tf01, RegisTf& tf12, RegisTf& tf02, double* opt_vars);
    void optimize(RegisTf& tf01, RegisTf& tf12, RegisTf& tf02, double* opt_vars, double* opt_sigs);
    // void update_windows_after_optimize(RegisTf& tf02, RegisTf& tf12);
    void set_frame_idx(int id) {
        curr_frame = id;
    };
};

#endif  // IMREG_FMT_OPTIMIZER_H
