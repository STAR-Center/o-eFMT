#include "image_registration.h"
#include <algorithm>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#define INF 10000000
#define minInThree(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
void dwt(Eigen::VectorXd& v, Eigen::VectorXd& w) {
    Eigen::MatrixXi mGamma;
    Eigen::MatrixXd mGamma_cont;
    int cost;
    for (int i = 1; i <= w.size(); i++) {
        mGamma(0, i) = INF;
    }
    for (int i = 1; i <= v.size(); i++) {
        mGamma(i, 0) = INF;
    }
    mGamma(0, 0) = 0;

    for (int i = 1; i <= v.size(); i++) {
        for (int j = 1; j <= w.size(); j++) {
            //	cost = abs( v[i] - w[j] ) * abs( v[i] - w[j] );
            if (v[i - 1] == w[j - 1]) {
                cost = 0;
                //	cout << v[i - 1] << " and " << w[j - 1] << endl;
            } else {
                cost = 1;
            }
            mGamma(i, j) = cost + minInThree(mGamma(i - 1, j), mGamma(i, j - 1), mGamma(i - 1, j - 1));
        }
    }
}

ImageRegistration::ImageRegistration(const cv::Mat& im)
    : rows_(im.rows), cols_(im.cols), log_polar_size_(std::max(rows_, cols_)), polar_size_(log_polar_size_), imdft_(rows_, cols_),
      imdft_logpolar_(log_polar_size_, log_polar_size_),
      image_transforms_(rows_, cols_, log_polar_size_, log_polar_size_, log_polar_size_, log_polar_size_) {
    high_pass_filter_ = imdft_.getHighPassFilter();
    initialize(im);
}

ImageRegistration::~ImageRegistration() {}

void ImageRegistration::initialize(const cv::Mat& im) {
    processImage(im, im0_gray_, im0_logpolar_);
}

void ImageRegistration::phaseCorrelate_rot_scale(const cv::Mat& im0,
                                                 const cv::Mat& im1,
                                                 double& rs_row,
                                                 double& rs_col,
                                                 double& rs_theta_sigma,
                                                 Eigen::VectorXd& scale_energy_vector,
                                                 bool base_version,
                                                 bool register01) {
    int register_id = frame_idx - (register01 ? 1 : 2);
    // imdft_logpolar_.phaseCorrelate(im1_logpolar_, im0_logpolar_, rs_row, rs_col, false, true);
    Eigen::MatrixXd abs = imdft_logpolar_.getPSD(im0, im1);
    // Utility::debug_output("rot-scale-spectrum" + std::to_string(register_id) + "_" + std::to_string(frame_idx), abs);
    // abs = abs.cwiseProduct()
    imdft_logpolar_.getPosAndVector(abs, rs_row, rs_col, rs_theta_sigma, scale_energy_vector, base_version);
    rs_row = rs_row - double(abs.rows() / 2);
    rs_col = rs_col - double(abs.cols() / 2);
    // Utility::debug_output("scale_energy_vector" + std::to_string(register_id) + "_" + std::to_string(frame_idx), scale_energy_vector);
}

void ImageRegistration::phaseCorrelate_translate(const cv::Mat& im0,
                                                 const cv::Mat& im1,
                                                 double& t_row,
                                                 double& t_col,
                                                 double& t_theta_sigma,
                                                 Eigen::VectorXd& trans_energy_vector,
                                                 bool base_version,
                                                 bool register01) {
    // check whether the store way will affect the cv output matrix
    // Done: It is OK.
    int register_id = frame_idx - (register01 ? 1 : 2);
    Eigen::MatrixXd abs = imdft_.getPSD(im0, im1);
    // Utility::debug_output("translation-spectrum" + std::to_string(register_id) + "_" + std::to_string(frame_idx), abs);
    cv::Mat cv_abs, cv_polar_abs;
    cv::eigen2cv(abs, cv_abs);
    image_transforms_.remapPolar(cv_abs, cv_polar_abs);
    cv::cv2eigen(cv_polar_abs, abs);
    // Utility::debug_output("translation-spectrum-polar" + std::to_string(register_id) + "_" + std::to_string(frame_idx), abs);

    imdft_.getPosAndVector(abs, t_row, t_col, t_theta_sigma, trans_energy_vector, base_version);
    // Utility::debug_output("trans_energy_vector" + std::to_string(register_id) + "_" + std::to_string(frame_idx), trans_energy_vector);
}

void ImageRegistration::registerImage(const cv::Mat& im,
                                      cv::Mat& registered_image,
                                      std::vector<double>& transform_params,
                                      bool display_images,
                                      bool base_version) {
    double rs_row, rs_col;
    double t_row, t_col;
    double scale = 1, rotation = 0;
    double delta_row, delta_col;
    double rs_theta_sigma = 0, t_theta_sigma = 0;

    processImage(im, im1_gray_, im1_logpolar_);

    phaseCorrelate_rot_scale(im1_logpolar_, im0_logpolar_, rs_row, rs_col, rs_theta_sigma, curr_scale_vec, base_version);
    image_transforms_.getScaleRotation(rs_row, rs_col, scale, rotation, rs_theta_sigma);
    image_transforms_.rotateAndScale(im0_gray_, im0_rotated_, 1 / scale, rotation);

    transform_params[2] = rotation;
    transform_params[3] = scale;

    if (display_images) {
        cv::imshow("im0_rotated", im0_rotated_);
        cv::imwrite("output/img/im0_rotated.tiff", im0_rotated_);
    }

    phaseCorrelate_translate(im1_gray_, im0_rotated_, t_row, t_col, t_theta_sigma, curr_trans_vec, base_version);
    image_transforms_.getTranslation(t_row, t_col, delta_row, delta_col, t_theta_sigma);
    image_transforms_.translate(im0_rotated_, registered_image, delta_col, delta_row);  // x, y

    transform_params[0] = delta_col;
    transform_params[1] = delta_row;

    if (display_images) {
        cv::imshow("im0_registered", registered_image);
        cv::imwrite("output/img/im0_registered.tiff", registered_image);
    }
}

void ImageRegistration::next() {
    im1_logpolar_.copyTo(im0_logpolar_);
    im1_gray_.copyTo(im0_gray_);
    last_scale_vec = curr_scale_vec;
    last_trans_vec = curr_trans_vec;
}

cv::Mat ImageRegistration::getBorderMask() {
    return image_transforms_.getBorderMask();
}

void ImageRegistration::processImage(const cv::Mat& im, cv::Mat& gray, cv::Mat& log_polar) {
    // why doesn't it convert to gray first and then float
    if (im.channels() == 3) {
        cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    } else {
        im.copyTo(gray);
    }
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    cv::Mat apodized;
    cv::Mat im_dft_cv;

    image_transforms_.apodize(gray, apodized);  // apodized = grapy.mul(appodizationWindow), smooth borders
    Eigen::MatrixXf im_dft = (imdft_.fftShift(imdft_.fft(apodized)).cwiseProduct(high_pass_filter_).cwiseAbs()).cast<float>();
    cv::eigen2cv(im_dft, im_dft_cv);                        // This function is really great!!!
    image_transforms_.remapLogPolar(im_dft_cv, log_polar);  // conver it to polar
}

cv::Mat ImageRegistration::getPreviousImage() {
    return im0_gray_;
}

cv::Mat ImageRegistration::getCurrentImage() {
    return im1_gray_;
}

// ================================================================
// ================= methods of Optimization class ================
// ================================================================
ImageRegistrationOpt::ImageRegistrationOpt(const cv::Mat& im0, const cv::Mat& im1, const double& fx_, const double& fy_) : ImageRegistration(im0) {
    processImage(im1, im1_gray_, im1_logpolar_);
    fx = fx_;
    fy = fy_;
}

cv::Mat ImageRegistrationOpt::getLastImage() {
    return im2_gray_;
}

/**
 * @brief img1 => im0, im2 => im1
 */
void ImageRegistrationOpt::next() {
    last_scale_vec = curr_scale_vec;
    last_trans_vec = curr_trans_vec;
    last_scale_sigma = curr_scale_sigma;
    last_trans_sigma = curr_trans_sigma;

    im1_logpolar_.copyTo(im0_logpolar_);
    im1_gray_.copyTo(im0_gray_);
    im2_logpolar_.copyTo(im1_logpolar_);
    im2_gray_.copyTo(im1_gray_);
}

void ImageRegistrationOpt::registerImage01(cv::Mat& registered_image, RegisTf& res, bool display_images, bool base_version, bool register01) {
    double rs_row, rs_col;
    double t_row, t_col;
    double scale, rotation;
    double delta_row, delta_col;
    double rs_theta_sigma, t_theta_sigma;

    phaseCorrelate_rot_scale(im1_logpolar_, im0_logpolar_, rs_row, rs_col, rs_theta_sigma, curr_scale_vec, base_version);
    image_transforms_.getScaleRotation(rs_row, rs_col, scale, rotation, rs_theta_sigma);
    image_transforms_.rotateAndScale(im0_gray_, im0_rotated_, scale, rotation);
    // Done: update curr_scale_sigma here

    res.theta = rotation / 180 * M_PI;
    res.scale = scale;
    res.sigma_theta = rs_theta_sigma;

    if (display_images) {
        cv::imshow("im01_rotated", im0_rotated_);
        cv::imwrite("output/imgs/im01_rotated.tiff", im0_rotated_);
    }

    phaseCorrelate_translate(im1_gray_, im0_rotated_, t_row, t_col, t_theta_sigma, curr_trans_vec, base_version);
    image_transforms_.getTranslation(t_row, t_col, delta_row, delta_col, t_theta_sigma);
    // Done: update curr_trans_sigma here
    curr_trans_sigma = t_theta_sigma;

    res.dx = delta_col;
    res.dy = delta_row;
    res.phi = atan2(res.dy, res.dx);  // [-pi, pi]
    res.sigma_phi = t_theta_sigma;

    image_transforms_.translate(im0_rotated_, registered_image, delta_col, delta_row);  // x, y
   if (display_images) {
       cv::imshow("im01_registered", registered_image);
       cv::imwrite("output/imgs/im01_registered.tiff", registered_image);
   }
}

/**
 * @brief register the img0 to img2, the output is stored in transform_params.
 * @param im2 input image
 * @param registered_image the registered image
 * @param transform_params register result (delta_col, delta_row, rotation, scale)
 * @param display_images whether show the image
 * @param base_version only select the maximum of spectrum rather than a vector
 */
void ImageRegistrationOpt::registerImage02(const cv::Mat& im2,
                                           cv::Mat& registered_image,
                                           RegisTf& res,
                                           bool display_images,
                                           bool base_version,
                                           bool register01) {
    double rs_row, rs_col;
    double t_row, t_col;
    double scale, rotation;
    double delta_row, delta_col;
    double rs_theta_sigma, t_theta_sigma;

    processImage(im2, im2_gray_, im2_logpolar_);

    phaseCorrelate_rot_scale(im2_logpolar_, im0_logpolar_, rs_row, rs_col, rs_theta_sigma, scale_vec_02, base_version, false);
    image_transforms_.getScaleRotation(rs_row, rs_col, scale, rotation, rs_theta_sigma);
    image_transforms_.rotateAndScale(im0_gray_, im02_rotated_, scale, rotation);
    // DONE: update scale_sigma_02 here
    // scale_sigma_02 = rs_theta_sigma;
    res.theta = rotation / 180 * M_PI;
    res.scale = scale;
    res.sigma_theta = rs_theta_sigma;

    //    if (display_images) {
    //        cv::imshow("im02_rotated", im02_rotated_);
    //        cv::imwrite("output/img/im02_rotated.tiff", im02_rotated_);
    //    }

    phaseCorrelate_translate(im2_gray_, im02_rotated_, t_row, t_col, t_theta_sigma, trans_vec_02, base_version, false);
    image_transforms_.getTranslation(t_row, t_col, delta_row, delta_col, t_theta_sigma);
    // Done: update trans_sigma_02 here
    trans_sigma_02 = t_theta_sigma;
    res.dx = delta_col;
    res.dy = delta_row;
    res.phi = atan2(res.dy, res.dx);  // [-pi, pi]
    res.sigma_phi = t_theta_sigma;

    image_transforms_.translate(im02_rotated_, registered_image, delta_col, delta_row);  // x, y
    //    if (display_images) {
    //        cv::imshow("im02_registered", registered_image);
    //        cv::imwrite("output/img/im02_registered.tiff", registered_image);
    //    }
}

/**
 * @brief Pattern matching for scale_energy_vector
 * @param last last scale_energy_vector
 * @param curr current scale_energy_vector
 * @param init the initial guess
 * @param lambda the result of Pattern matching
 * @param sigma the uncertainty of Pattern matching
 */
/* void ImageRegistrationOpt::PatternMatching_scale(const Eigen::VectorXd &last, const Eigen::VectorXd &curr,
                                                 const double &init, double &lambda, double &sigma) {
    double err_sum = 0;
//    std::vector<double> err_vec;
    // shift from -4 to 4, interval is 0.01
    double all_sum = 0;
    double min_val = 1e10;
    double shift = 0;
    int cnt = 0;
    for (int i = 0; i < 2 * 4 * 1000; ++i) {
        shift = -4 + i * 0.001;
        cnt = 0;
        err_sum = 0;
        for (int j = 0;
             j < int(last.size() * 2 / 3); ++j) {  // calculate scale with the far plane that the index is small.
            double cmp_j = j + shift;
            int cmp_j_floor = int(cmp_j);
            int cmp_j_ceil = cmp_j_floor + 1;
            if (cmp_j_floor < 0 || cmp_j_ceil >= int(curr.size() * 2 / 3)) continue;
            err_sum += abs(
                    last[j] - curr[cmp_j_floor] * (cmp_j - cmp_j_floor) + curr[cmp_j_ceil] * (cmp_j_ceil - cmp_j));
            cnt += 1;
        }
        err_sum /= cnt;
//        err_vec.push_back(err_sum);
        all_sum += err_sum;
        if (min_val > err_sum) {
            min_val = err_sum;
            lambda = shift;
        }
    }
    sigma = (all_sum - min_val) / all_sum;
    SPDLOG_LOGGER_DEBUG(Utility::logger, "PM of scale: init = {0}, lambda = {1}, sigma = {2}", init, lambda, sigma);
//    lambda = init;
//    sigma = 0.0;
} */

/**
 * @brief Pattern matching for translation energy vector
 * @param last last translation energy vector
 * @param curr current translation energy vector
 * @param init the initial guess
 * @param delta movement through z-axis
 * @param zp the z of last frame
 * @param lambda the pattern matching result
 * @param sigma the uncertainty of pattern matching
 */
void ImageRegistrationOpt::PatternMatching_trans(const Eigen::VectorXd& last,
                                                 const Eigen::VectorXd& curr,
                                                 const double& init,
                                                 const double& delta,
                                                 const double& zp,
                                                 double& lambda,
                                                 double& sigma) {
    //    basically the depth(usually more than 20 m) is rather further than the move distance(usually less than 1 m), so the scale is around 1.
    int dt_pi;
    double max = last.maxCoeff(&dt_pi);
    double err_sum = 0;
    double min_val = 1e10;
    double all_sum = 0;
    double scale = 0;
    for (int i = 1; i < 100; ++i) {
        scale = 1 - 0.6 / i;
        int cnt = 0;
        err_sum = 0;
        for (int j = 0; j < last.size(); ++j) {
            double cmp_j = j * scale * (dt_pi * zp) / (dt_pi * zp + j * delta);
            int cmp_j_floor = int(cmp_j);
            int cmp_j_ceil = cmp_j_floor + 1;
            if (cmp_j_floor < 0 || cmp_j_ceil >= curr.size() * 2 / 3)
                continue;
            err_sum += abs(last[j] - curr[cmp_j_floor] * (cmp_j - cmp_j_floor) + curr[cmp_j_ceil] * (cmp_j_ceil - cmp_j));
            cnt += 1;
        }
        err_sum /= cnt;
        all_sum += err_sum;
        if (min_val > err_sum) {
            min_val = err_sum;
            lambda = scale;
        }
    }
    for (int i = 1; i < 100; ++i) {
        scale = 1 - 0.6 / i;
        int cnt = 0;
        err_sum = 0;
        for (int j = 0; j < curr.size(); ++j) {
            double cmp_j = j * scale * (dt_pi * zp + j * delta) / (dt_pi * zp);
            int cmp_j_floor = int(cmp_j);
            int cmp_j_ceil = cmp_j_floor + 1;
            if (cmp_j_floor < 0 || cmp_j_ceil >= last.size() * 2 / 3)
                continue;
            err_sum += abs(curr[j] - last[cmp_j_floor] * (cmp_j - cmp_j_floor) + last[cmp_j_ceil] * (cmp_j_ceil - cmp_j));
            cnt += 1;
        }
        err_sum /= cnt;
        all_sum += err_sum;
        if (min_val > err_sum) {
            min_val = err_sum;
            lambda = 1 / scale;
        }
    }
    sigma = (all_sum - min_val) / all_sum;
    SPDLOG_LOGGER_DEBUG(Utility::logger, "PM of translation: init = {0}, lambda = {1}, sigma = {2}", init, lambda, sigma);
    //    lambda = init;
    //    sigma = 0.0;
}

void scaleInterpolateVec(const Eigen::VectorXd& v1, double scale, Eigen::VectorXd& res) {
    double idx, nidx;
    int upper, lower;
    res.resize(v1.size());
    for (int i = 0; i < v1.size() - 1; i++) {
        // scale should be not less than 1.0
        idx = i * scale;
        nidx = (i + 1) * scale;
        if (int(nidx) >= res.size()) {
            nidx = res.size() - 1;
        }
        // std::cout << "i, scale, st, ed = "<< i << "\t" << scale << "\t" << ceil(idx) <<"\t" << int(nidx) << std::endl;
        for (int j = ceil(idx); j <= int(nidx); j++) {
            res[j] = v1[i] + (v1[i + 1] - v1[i]) * (j - idx) / (nidx - idx);
        }
    }
}

bool sort_by_second(const std::pair<int, int>& a, const std::pair<int, int>& b) {
    return a.second > b.second;
}

// PatternMatching trans&scale energy vector
void ImageRegistrationOpt::PatternMatching_trans_scale(const Eigen::VectorXd& trans,
                                                       const Eigen::VectorXd& scale,
                                                       const double& init,
                                                       double& lambda,
                                                       double& sigma,
                                                       std::string debug_name,
                                                       int key) {
    double lmd = 0;
    double err_sum = 0;
    double min_err = 1e10;
    double err = 0;
    int num = 100;

    Eigen::MatrixXd scale_factor(num * 2 + 1, 2);
    scale_factor(num, 0) = 1.0;
    for (int i = num + 1; i < 2 * num + 1; i++) {
        // scale_factor(i,0) = 4 / (i + 1) + 0.6; // hyperparameters
        scale_factor(i, 0) = exp((i - num) * 1.0 / num * (key / 10.0));  // hyperparameters
    }
    for (int i = 0; i < num; i++) {
        scale_factor(i, 0) = 1.0 / scale_factor(scale_factor.rows() - 1 - i, 0);
    }
    double scale_peak_pos, trans_peak_pos;
    scale.maxCoeff(&scale_peak_pos);
    trans.maxCoeff(&trans_peak_pos);
    double max_s_factor = scale.size() / (scale_peak_pos + 2);
    double max_t_factor = trans.size() / (trans_peak_pos + 2);
    double max_err = 0;
    int valid_cnt = 0;
    int valid_st_idx = 0, valid_ed_idx = scale_factor.rows() - 1;
    Eigen::VectorXd v1, v2;
    for (int i = 0; i < num * 2 + 1; i++) {
        double curr_scale = scale_factor(i, 0);
        if (curr_scale < 1.) {
            if (scale_factor(scale_factor.rows() - 1 - i, 0) >= max_s_factor) {
                scale_factor(i, 1) = 1e6;
                valid_st_idx = i;
                continue;
            }
            scaleInterpolateVec(scale, scale_factor(scale_factor.rows() - 1 - i, 0), v1);
            v2 = trans;
        } else {
            if (curr_scale >= max_t_factor) {
                scale_factor(i, 1) = 1e6;
                valid_ed_idx = i;
                break;
            }
            scaleInterpolateVec(trans, curr_scale, v1);
            v2 = scale;
        }

        // divide by mean
        double v1_mean = v1.mean();
        double v2_mean = v2.mean();
        err = 0;
        for (int j = 0; j < v1.size(); j++) {
            v1[j] /= v1_mean;
            v2[j] /= v2_mean;
            err += pow(v2[j] - v1[j], 2);
        }

        max_err = max_err < err ? err : max_err;
        scale_factor(i, 1) = err;
        err_sum += err;
    }
    valid_st_idx++;
    valid_cnt = valid_ed_idx - valid_st_idx;
    // std::sort(scale_factor.begin(), scale_factor.end(), sort_by_second);

    // double maxi, mini;
    // scale_factor.col(1).maxCoeff(&maxi);
    // scale_factor.col(1).minCoeff(&mini);

    // double weighted_idx_sum = 0;
    // double weight_sum = 0;
    // double weighted_result = -1;
    // double mmin = maxi + 1;
    double avg = 0;
    /* for (int i = 0; i < scale_factor.rows(); i++) {
        if (scale_factor(i, 0) < 0.55) {
            scale_factor(i, 1) = maxi;
        }
        // scale_factor(i, 1) = (scale_factor(i, 1) - mini) / (maxi - mini);

        // (i, 0) is index, (i, 1) is value
        if (scale_factor(i, 1) < mini * 1.5 && init != -1) {
            weighted_idx_sum += 1 / (scale_factor(i, 1) + 0.001) * scale_factor(i, 0);  // sum of the weighted value
            weight_sum += 1 / (scale_factor(i, 1) + 0.001);                             // sum of the weight
            mmin = mmin > scale_factor(i, 1) ? scale_factor(i, 0) : mmin;

            if (i + 1 >= scale_factor.rows() || scale_factor(i + 1, 1) >= 0.15) {
                weighted_result = weighted_idx_sum / weight_sum;
                lambda = (std::abs(weighted_result - init) < lambda) ? std::abs(weighted_result) : lambda;
                // lambda = (std::abs(scale_factor(i,0) - init) < lambda) ? std::abs(scale_factor(i,0)) : lambda;
                sigma = weighted_idx_sum * mmin * 1.5;

                weight_sum = 0;
                weighted_idx_sum = 0;
                weighted_result = -1;
            }
        }
    } */

    scale_factor.col(1).segment(0, valid_st_idx).setOnes();
    scale_factor.col(1).segment(0, valid_st_idx) *= scale_factor(valid_st_idx, 1);
    scale_factor.col(1).segment(valid_ed_idx, scale_factor.rows() - valid_ed_idx).setOnes();
    scale_factor.col(1).segment(valid_ed_idx, scale_factor.rows() - valid_ed_idx) *= scale_factor(valid_ed_idx - 1, 1);
    int min_idx = 0;
    scale_factor.col(1).segment(valid_st_idx, valid_cnt).minCoeff(&min_idx);
    min_idx += valid_st_idx;
    lambda = scale_factor(min_idx, 0);
    min_err = scale_factor(min_idx, 1);
    std::cout << "valid_st=" << valid_st_idx << ", valid_ed=" << valid_ed_idx << ", (" << valid_cnt << "). PM result, take the min_err " << min_err
              << ", the lambda is " << lambda << std::endl;
    SPDLOG_LOGGER_DEBUG(Utility::logger, "valid_st= {0}, valid_ed= {1}, ({2}). PM result: the lambda is {4}, take the min_err= {4}", valid_st_idx,
                        valid_ed_idx, lambda, min_err);

    // Laplace filter and find the peak, which is the valley in errs.
    int kernel_sz = 3;
    Eigen::VectorXd laplace_kernel(kernel_sz);
    // laplace_kernel << 1,4,-10,4,1;
    laplace_kernel << 2., -4., 2.;
    Eigen::VectorXd errs = scale_factor.col(1);
    for (int i = kernel_sz / 2; i < errs.size() - kernel_sz / 2; i++) {
        errs(i) = scale_factor.col(1).segment(i - kernel_sz / 2, kernel_sz).dot(laplace_kernel);
    }

    int max_laplace_idx = valid_st_idx;
    for (int i = valid_st_idx; i < valid_st_idx + valid_cnt; i++) {
        if ((scale_factor(i, 1) - min_err) / (max_err - min_err) < 0.22 && (std::abs(scale_factor(i, 0) - init) < 0.6 || init < 0)) {
            max_laplace_idx = errs(i) > errs(max_laplace_idx) ? i : max_laplace_idx;
        }
    }

    // errs.segment(valid_st_idx, valid_cnt).maxCoeff(&max_laplace_idx);
    // max_laplace_idx += valid_st_idx;
    min_err = scale_factor(max_laplace_idx, 1);
    if (max_laplace_idx - 2 >= valid_st_idx){
        max_laplace_idx -= 2;
        // lambda = (scale_factor(max_laplace_idx, 0) + scale_factor(max_laplace_idx+1, 0)) / 2;
    }
    lambda = scale_factor(max_laplace_idx, 0);

    avg = err_sum / valid_cnt;
    sigma = (min_err / avg) * 0.9 + 0.01;  // [0.05, 0.95]
    // Utility::debug_output(debug_name, scale_factor);
    SPDLOG_LOGGER_DEBUG(Utility::logger, "PMT: scale,min_err,sigma,err_sum,avg,max_laplace_idx = {0}\t{1}\t{2}\t{3}\t{4}\t{5}", lambda, min_err,
                        sigma, err_sum, avg, max_laplace_idx);
}

void ImageRegistrationOpt::PatternMatching_trans(const Eigen::VectorXd& last,
                                                 const Eigen::VectorXd& curr,
                                                 const double& init,
                                                 double& lambda,
                                                 double& sigma,
                                                 std::string debug_name,
                                                 int key) {
    // Eigen::VectorXd sorted_last = last;
    // Eigen::VectorXd sorted_curr = curr;
    // std::sort(sorted_last.data(), sorted_last.data() + sorted_last.size(), std::greater<double>());
    // std::sort(sorted_curr.data(), sorted_curr.data() + sorted_curr.size(), std::greater<double>());
    // double header = sorted_last.segment(0, 5).sum();
    // double mean = (sorted_last.sum() - header) / (sorted_last.size() - 5);
    // std::cout << "header " << header << " / mean " << mean << " = " << header / mean << std::endl;
    PatternMatching_trans_scale(last, curr, init, lambda, sigma, debug_name, key);
}

void ImageRegistrationOpt::PatternMatching_scale(const Eigen::VectorXd& last,
                                                 const Eigen::VectorXd& curr,
                                                 const double& init,
                                                 double& lambda,
                                                 double& sigma,
                                                 std::string debug_name) {
    double lmd, err_sum;
    double min_err = 1e10;
    double err = 0;
    int num = last.size() / 10 * 25 + 1;
    Eigen::MatrixXd scale_factor(num, 2);
    for (int i = 0; i < num; i++) {
        lmd = (i - num / 2.0) / 25;  // hyperparameters
        err = 0;
        double curr_idx;
        int c_idx_lower, c_idx_upper;
        int cnt = 0;

        for (int i = 0; i < last.size(); i++) {
            curr_idx = i + lmd;
            c_idx_lower = int(curr_idx);
            c_idx_upper = c_idx_lower + 1;
            if (c_idx_lower >= 0 && c_idx_upper < curr.size()) {
                err += pow((curr_idx - c_idx_lower) * curr[c_idx_lower] + (c_idx_upper - curr_idx) * curr[c_idx_upper] - last[i], 2);
                cnt++;
            }
        }
        err /= (cnt * cnt);
        scale_factor(i, 0) = lmd;
        scale_factor(i, 1) = err;
        err_sum += err;
        if (min_err > err) {
            min_err = err;
            lambda = lmd;
        }
    }

    double avg = err_sum / num;
    sigma = (min_err / avg) * 0.9 + 0.01;  // [0.05, 0.95]
    // Utility::debug_output(debug_name, scale_factor);
    SPDLOG_LOGGER_DEBUG(Utility::logger, "PMS: scale,min_err,sigma,err_sum,avg = {0}\t{1}\t{2}\t{3}\t{4}", lambda, min_err, sigma, err_sum, avg);
}

// TODO: Bug Here, the theta should be in radain rather degree
void ImageRegistrationOpt::get_transform(const RegisTf& last_res,
                                         const Pose& last_pose,
                                         RegisTf& curr_res,
                                         Pose& curr_pose,
                                         const Eigen::VectorXd& trans_vec,
                                         const Eigen::VectorXd& scale_vec,
                                         const Eigen::VectorXd& last_trans_vec,
                                         const Eigen::VectorXd& last_scale_vec,
                                         const bool& interval_2,
                                         const Pose& base_pose) {
    double lambda_s_sigma = 0, lambda_t_sigma = 0;
    // Pattern matching for scale energy vector
    PatternMatching_scale(last_scale_vec, scale_vec, 1.0, curr_lambda_s, lambda_s_sigma);
    double scale = last_res.scale / pow(image_transforms_.getLogbase(), curr_lambda_s);
    curr_res.scale = (curr_res.scale + scale) / 2;
    curr_res.sigma_scale = log(image_transforms_.getLogbase()) / pow(image_transforms_.getLogbase(), lambda_s_sigma) * 0.5 * last_res.scale;
    if (interval_2) {
        curr_pose.z = base_pose.z / curr_res.scale;
        curr_res.dz = curr_pose.z - base_pose.z;
    } else {
        curr_pose.z = last_pose.z / curr_res.scale;
        curr_res.dz = curr_pose.z - last_pose.z;
    }
    // TODO: uncertainty of z
    curr_res.sigma_z = 2 * last_res.scale * last_pose.z * pow(image_transforms_.getLogbase(), lambda_s_sigma) * log(image_transforms_.getLogbase()) /
                       pow((last_res.scale + curr_res.scale * pow(image_transforms_.getLogbase(), lambda_s_sigma)), 2);
    // Pattern matching for translation energy vector
    PatternMatching_trans(last_trans_vec, trans_vec,
                          sqrt((curr_res.dx * curr_res.dx + curr_res.dy * curr_res.dy) / (last_res.dx * last_res.dx + last_res.dy * last_res.dy)),
                          curr_res.dz, last_pose.z, curr_lambda_t, lambda_t_sigma);
    // TODO: uncertainty of x,y
    // curr_res.sigma_x = lambda_t_sigma * cos(curr_res.theta) * 2;
    // curr_res.sigma_y = lambda_t_sigma * sin(curr_res.theta) * 2;
    // update the current pose
    if (interval_2) {
        curr_pose.x = ((-1 * curr_pose.z * curr_res.dx / fx + base_pose.x) + (curr_lambda_t * last_pose.x)) / 2;
        curr_pose.y = ((-1 * curr_pose.z * curr_res.dy / fy + base_pose.y) + (curr_lambda_t * last_pose.y)) / 2;
    } else {
        curr_pose.x = ((-1 * curr_pose.z * curr_res.dx / fx + last_pose.x) + (curr_lambda_t * last_pose.x)) / 2;
        curr_pose.y = ((-1 * curr_pose.z * curr_res.dy / fy + last_pose.y) + (curr_lambda_t * last_pose.y)) / 2;
    }
    curr_pose.yaw = fmod((last_pose.yaw + curr_res.theta / 180 * M_PI), (M_PI * 2)) - M_PI;
    SPDLOG_LOGGER_DEBUG(Utility::logger, "registerTF(dx,dy,dz,theta,scale): ({0}, {1}, {2}, {3}, {4}), lambda_scale={5}, lambda_trans={6}",
                        curr_res.dx, curr_res.dy, curr_res.dz, curr_res.theta, curr_res.scale, curr_lambda_s, curr_lambda_t);
}

void ImageRegistrationOpt::get_transform(const RegisTf& res01, RegisTf& res12, RegisTf& res02, double* opt_vars, double* opt_sigs) {
    // as link chain rule
    double PMT01_12, PMT01_02, sig01_12, sig01_02, PMT12_02, sig12_02;
    PatternMatching_trans(last_trans_vec, curr_trans_vec, 1., PMT01_12, sig01_12, "PMT01_12_" + std::to_string(frame_idx), 16);
    PatternMatching_trans(last_trans_vec, trans_vec_02, 2., PMT01_02, sig01_02, "PMT01_02_" + std::to_string(frame_idx), 13);
    PatternMatching_trans(curr_trans_vec, trans_vec_02, 2., PMT12_02, sig12_02, "PMT12_02_" + std::to_string(frame_idx), 13);
    PatternMatching_scale(last_scale_vec, curr_scale_vec, 0, opt_vars[2], opt_sigs[2], "PMS01_12_" + std::to_string(frame_idx));
    PatternMatching_scale(last_scale_vec, scale_vec_02, 0, opt_vars[3], opt_sigs[3], "PMS01_02_" + std::to_string(frame_idx));
    PatternMatching_trans_scale(trans_vec_02, scale_vec_02, -1, opt_vars[4], opt_sigs[4], "PMST01_02_" + std::to_string(frame_idx), 12);
    PatternMatching_trans_scale(curr_trans_vec, curr_scale_vec, -1, opt_vars[5], opt_sigs[5], "PMST01_12_" + std::to_string(frame_idx), 12);

    opt_vars[0] = PMT01_12;
    opt_vars[1] = PMT01_02;
    opt_sigs[0] = sig01_12;
    opt_sigs[1] = sig01_02;
    // if (std::abs(PMT12_02 - 2) < std::abs(PMT01_02 - 2)) {
    //     opt_vars[1] = PMT12_02;
    //     opt_sigs[1] = std::sqrt(sig12_02 * sig12_02 + sig01_02 * sig01_02);
    // }
    // if (std::abs(PMT01_12 - 1) > 0.5 & frame_idx > 2) {
    //     opt_vars[0] = 1.;
    //     opt_sigs[0] = 0.01;
    // }

    opt_vars[6] = res12.phi;
    opt_vars[7] = res02.phi;
    opt_vars[8] = res12.theta;
    opt_vars[9] = res02.theta;
    opt_sigs[6] = res12.sigma_phi;
    opt_sigs[7] = res02.sigma_phi;
    opt_sigs[8] = res12.sigma_theta;
    opt_sigs[9] = res02.sigma_theta;

    // X(8 dim): PMT01_12, PMT01_02, PMS01_12, PMS01_12, PMST12, PMST02, phi12, phi02, theta12, theta02
}

void ImageRegistrationOpt::init_base(const Pose& pose0, Pose& pose1, RegisTf& first_tf) const {
    pose1.z = pose0.z / first_tf.scale;
    first_tf.dz = pose1.z - pose0.z;
    pose1.x = pose0.x + (pose1.z * first_tf.dx / fx);
    pose1.y = pose0.y + (pose1.z * first_tf.dy / fy);
    pose1.yaw = fmod(pose0.yaw + first_tf.theta, 2 * M_PI) - M_PI;
}

void ImageRegistrationOpt::init_base(RegisTf& first_tf) {
    double lambda = 1.0, sigma = 0;
    PatternMatching_trans_scale(curr_trans_vec, curr_scale_vec, 0.0, lambda, sigma);
    // first_tf.dz = lambda * 1.0;  // optimize z with translation on XOY
    first_tf.dz = first_tf.zp / first_tf.scale - first_tf.zp; 
    first_tf.zl = first_tf.zp = first_tf.dz;
}

ImageRegistrationOpt::~ImageRegistrationOpt() = default;
