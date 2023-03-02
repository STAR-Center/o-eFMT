#include "optimizer.h"
#include <iostream>

// X(10 dim): PMT01_12, PMT01_02, PMS01_12, PMS01_12, PMST12, PMST02, phi12, phi02, theta12, theta02
// err_lambda = lambda01 + PMT01_12*lambda01 - PMT01_02*lambda01;
// err_z = z01 + z12 - z02
//       = z01 + (PMST12*PMT01_12*lambda01 + PMS01_12+z01) - (PMST02*PMT01_02*lambda01 + PMS01_02+z01)
// err_theta = ((theta01 + theta12)%(2pi) - pi) - theta02
struct CostFunctor {
    CostFunctor(double lambda01, double z01, double phi01, double theta01, double scale01) : lambda01(lambda01), z01(z01), phi01(phi01), theta01(theta01), scale01(scale01) {}
    template <typename T> bool operator()(const T* const X, T* residual) const {
        residual[0] = X[0] * lambda01 * cos(theta01 + X[6] + X[8]) + lambda01 * cos(theta01 + phi01) - X[1] * lambda01 * cos(X[7] + X[9]);
        residual[1] = X[0] * lambda01 * sin(theta01 + X[6] + X[8]) + lambda01 * sin(theta01 + phi01) - X[1] * lambda01 * sin(X[7] + X[9]);
        // residual[2] = z01 + (X[4] * X[0] * lambda01 + X[2] + z01) * 0.5 - (X[5] * X[0] * lambda01 + X[3] + z01) * 0.5;
        residual[2] = scale01 * (X[3] / X[2]) - 1.0;
        return true;
    }
    double lambda01, z01, phi01, theta01, scale01;
};

// void Optimizer::check_before_optimize(RegisTf& tf01, RegisTf& tf12, RegisTf& tf02, double* opt_vars){
//     if (curr_frame - 2 > windows_sz) {
//         double mean_lambda = sqrt((mean.dx / windows_sz) * (mean.dx / windows_sz) + (mean.dy / windows_sz) * (mean.dy / windows_sz));
//         double tf12_lambda = sqrt(tf12.dx * tf12.dx + tf12.dy * tf12.dy);
//         double tf01_lambda = sqrt(tf01.dx * tf01.dx + tf01.dy * tf01.dy);
//         double tf02_lambda = sqrt(tf02.dx * tf02.dx + tf02.dy * tf02.dy);
//         if (tf12_lambda / mean_lambda > 1.15 || mean_lambda/tf12_lambda > 1.15) {
//             tf12.dx = mean_lambda * cos(tf12.phi);
//             tf12.dy = mean_lambda * sin(tf12.phi);
//             if (abs(opt_vars[0] - 1.0) > 0.15) {
//                 opt_vars[0] = 1.0;
//             }
//         }
//         double x02, y02;
//         x02 = opt_vars[0] * tf01_lambda * cos(tf01.theta + opt_vars[6] + opt_vars[8]) + tf01_lambda * cos(tf01.theta + tf01.phi);
//         y02 = opt_vars[0] * tf01_lambda * sin(tf01.theta + opt_vars[6] + opt_vars[8]) + tf01_lambda * sin(tf01.theta + tf01.phi);
//         double tf02_l = sqrt(x02*x02 + y02*y02);
//         if(abs(tf02_lambda-tf02_l) / tf02_l > 0.5 ){
//             opt_vars[1] = tf02_l / tf01_lambda;
//             tf02.dx = tf02_l * cos(tf02.phi);
//             tf02.dy = tf02_l * sin(tf02.phi);
//         }
//         std::cout << "mean: "<< mean.dx << ", " << mean.dy << ", tf01: "<< tf01.dx << ", " << tf01.dy << ", tf12: " <<  tf12.dx << ", " << tf01.dy << ", tf02: " <<  tf02.dx << ", " << tf02.dy << std::endl;
//     }
// }

// void Optimizer::update_windows_after_optimize(RegisTf& tf01, RegisTf& tf12){
//     if ((curr_frame - 2) > windows_sz) {
//         // double tf12_lambda = sqrt(tf12.dx * tf12.dx + tf12.dy * tf12.dy);
//         // double mean_lambda = sqrt((mean.dx / windows_sz) * (mean.dx / windows_sz) + (mean.dy / windows_sz) * (mean.dy / windows_sz));
//         // if (tf12_lambda / mean_lambda > 1.2 || mean_lambda/tf12_lambda > 1.2) {
//         //     tf12.dx = (tf12_lambda+mean_lambda) / 2 * cos(tf12.phi);
//         //     tf12.dy = (tf12_lambda+mean_lambda) / 2 * sin(tf12.phi);
//         // }
//         RegisTf tmp = windows.front();
//         mean.dx -= tmp.dx;
//         mean.dy -= tmp.dy;
//         windows.pop();
//     }
//     mean.dx += tf12.dx;
//     mean.dy += tf12.dy;
//     windows.push(tf12);
// }


void Optimizer::optimize(RegisTf& tf01, RegisTf& tf12, RegisTf& tf02, double* opt_vars, double* opt_sigs) {
    std::cout << "Ceres optimization begin ... " << std::endl;
    double lambda01 = sqrt(tf01.dx * tf01.dx + tf01.dy * tf01.dy);

    ceres::Problem problem;
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CostFunctor, 3, 10>(new CostFunctor(lambda01, tf01.dz, tf01.phi, tf01.theta, tf01.scale)),
                             // nullptr,
                             new ceres::SoftLOneLoss(0.5),  // lose_function
                             // new ceres::HuberLoss(1.0),  // kernel fucntion
                             opt_vars);
    for (int i = 0; i < 6; i++) {
        problem.SetParameterLowerBound(opt_vars, i, opt_vars[i] - abs(opt_vars[i]) * opt_sigs[i]);
        problem.SetParameterUpperBound(opt_vars, i, opt_vars[i] + abs(opt_vars[i]) * opt_sigs[i]);
    }
    for (int i = 6; i < 10; i++) {
        problem.SetParameterLowerBound(opt_vars, i, opt_vars[i] - opt_sigs[i]);
        problem.SetParameterUpperBound(opt_vars, i, opt_vars[i] + opt_sigs[i]);
    }

    // show some information here ...
    std::cout << "Solving ceres optimization ... " << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    // problem.~Problem();

    // check the tf12 and tf02, if it is too strange, set them to the average of 5 historic value.
    // if the lambda is so strange, set the value to 

    // TODO: recompute tf12 and tf02 according to opt_var
    // X(8 dim): PMT01_12, PMT01_02, PMS01_12, PMS01_12, PMST12, PMST02, theta12, theta02
    if (curr_frame -2 >= windows_sz){
        double mean_lambda = sum_lambda / windows_sz;
        double tf01_lambda = sqrt(tf01.dx * tf01.dx + tf01.dy * tf01.dy);
        double tf12_lambda = sqrt(tf12.dx * tf12.dx + tf12.dy * tf12.dy);
        if (tf12_lambda / mean_lambda > 1.25 || tf12_lambda / mean_lambda < 0.75) {
            tf12.dx = mean_lambda * cos(tf12.phi);
            tf12.dy = mean_lambda * sin(tf12.phi);
        }
        double tf02_lambda = sqrt(tf02.dx * tf02.dx + tf02.dy * tf02.dy);
        double x02, y02;
        x02 = opt_vars[0] * tf01_lambda * cos(tf01.theta + opt_vars[6] + opt_vars[8]) + tf01_lambda * cos(tf01.theta + tf01.phi);
        y02 = opt_vars[0] * tf01_lambda * sin(tf01.theta + opt_vars[6] + opt_vars[8]) + tf01_lambda * sin(tf01.theta + tf01.phi);
        double tf02_l = sqrt(x02*x02 + y02*y02);
        if(abs(tf02_lambda-tf02_l) / tf02_l > 0.5 ){
            tf02.dx = tf02_l * cos(tf02.phi);
            tf02.dy = tf02_l * sin(tf02.phi);
        }
    }

    if ((curr_frame - 2) >= windows_sz) {
        RegisTf tmp = windows.front();
        sum_lambda -= sqrt(tmp.dx * tmp.dx + tmp.dy * tmp.dy);
        windows.pop();
    }
    sum_lambda += sqrt(tf12.dx * tf12.dx + tf12.dy * tf12.dy);
    windows.push(tf12);
   
    // double lambda12 = opt_vars[0] * lambda01;
    // tf12.phi = opt_vars[6];
    // tf12.theta = opt_vars[8];
    // tf12.dx = lambda12 * cos(tf12.theta + tf12.phi);
    // tf12.dy = lambda12 * sin(tf12.theta + tf12.phi);
    // tf12.dz = (opt_vars[4] * lambda12 + (opt_vars[2] + tf01.dz)) / 2;
    // double lambda02 = opt_vars[1] * lambda01;
    // tf02.phi = opt_vars[7];
    // tf02.theta = opt_vars[9];
    // tf02.dx = lambda02 * cos(tf02.theta + tf02.phi);
    // tf02.dy = lambda02 * sin(tf02.theta + tf02.phi);
    // tf02.dz = (opt_vars[5] * lambda12 + (opt_vars[3] + tf01.dz)) / 2;

    double lambda12 = opt_vars[0] * lambda01;
    tf12.phi = (opt_vars[6] + tf12.phi)/2;
    tf12.theta = (opt_vars[8] + tf12.theta) / 2;
    tf12.dx = (lambda12 * cos(tf12.theta + tf12.phi) + tf12.dx) / 2;
    tf12.dy = (lambda12 * sin(tf12.theta + tf12.phi) + tf12.dy) / 2;
    tf12.scale = opt_vars[2];
    // tf12.dz = (opt_vars[4] * lambda12 + (opt_vars[2] + tf01.dz)) / 2;
    double lambda02 = opt_vars[1] * lambda01;
    tf02.phi = (opt_vars[7] + tf02.phi) / 2;
    tf02.theta = (opt_vars[9] + tf02.theta) / 2;
    tf02.dx = (lambda02 * cos(tf02.theta + tf02.phi) + tf02.dx) / 2;
    tf02.dy = (lambda02 * sin(tf02.theta + tf02.phi) + tf02.dy) / 2;
    // tf02.dz = (opt_vars[5] * lambda12 + (opt_vars[3] + tf01.dz)) / 2;
    tf02.scale = opt_vars[3];
}
