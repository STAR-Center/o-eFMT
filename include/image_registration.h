#ifndef IMAGE_REGISTRATION_H
#define IMAGE_REGISTRATION_H

//#include <spdlog/spdlog.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "image_dft.h"
#include "image_transforms.h"
#include "utility.h"

extern int frame_idx;

class ImageRegistration {
  public:
    ImageRegistration(const cv::Mat& im);

    virtual ~ImageRegistration();

    void initialize(const cv::Mat& im);

    /*
     * convert to grayscale
     * change range of pixel intensity to 0-1
     * smooth borders
     * get log-polar transform of DFT
     */
    void processImage(const cv::Mat& im, cv::Mat& gray, cv::Mat& log_polar);

    /*
     * register im to previous image (from previous call to the same function or initialize function)
     */
    void registerImage(const cv::Mat& im, cv::Mat& out, std::vector<double>& transform_params, bool display_images = false, bool base_version = true);

    void phaseCorrelate_rot_scale(const cv::Mat& im1,
                                  const cv::Mat& im0,
                                  double& rs_row,
                                  double& rs_col,
                                  double& row_sigma,
                                  Eigen::VectorXd& energy_vector,
                                  bool base_version = true,
                                  bool register01 = true);

    void phaseCorrelate_translate(const cv::Mat& im1,
                                  const cv::Mat& im0,
                                  double& rs_row,
                                  double& rs_col,
                                  double& row_sigma,
                                  Eigen::VectorXd& energy_vector,
                                  bool base_version = true,
                                  bool register01 = true);

    /*
     * return white image warped by same amount as last registration operation
     */
    cv::Mat getBorderMask();

    /*
     * prepare for next image by copying reference image
     *
     * this is not done in registerImage since the current and previous image
     * might be required after registration.
     */
    virtual void next();

    cv::Mat getPreviousImage();

    cv::Mat getCurrentImage();

    Eigen::VectorXd curr_scale_vec;
    Eigen::VectorXd curr_trans_vec;
    Eigen::VectorXd last_scale_vec;
    Eigen::VectorXd last_trans_vec;

  protected:
    int rows_;
    int cols_;
    int log_polar_size_;
    int polar_size_;

    ImageDFT imdft_;
    ImageDFT imdft_logpolar_;
    ImageTransforms image_transforms_;

    Eigen::MatrixXd high_pass_filter_;

    cv::Mat im0_gray_;
    cv::Mat im1_gray_;
    cv::Mat im0_logpolar_;
    cv::Mat im1_logpolar_;
    cv::Mat im0_rotated_;
};

class ImageRegistrationOpt : public ImageRegistration {
  public:
    explicit ImageRegistrationOpt(const cv::Mat& im0, const cv::Mat& im1, const double& fx_, const double& fy_);

    ~ImageRegistrationOpt() override;

    void next() override;

    //    void registerImage01(cv::Mat &out, std::vector<double> &transform_params, bool display_images = false, bool base_version = true);
    //    void registerImage02(const cv::Mat &im2, cv::Mat &out, std::vector<double> &transform_params, bool display_images = false, bool base_version
    //    = true);
    void registerImage01(cv::Mat& out, RegisTf& res, bool display_images = false, bool base_version = true, bool register01 = false);

    void
    registerImage02(const cv::Mat& im2, cv::Mat& out, RegisTf& res, bool display_images = false, bool base_version = true, bool register01 = false);

    cv::Mat getLastImage();

    Eigen::VectorXd scale_vec_02;
    Eigen::VectorXd trans_vec_02;

    void init_base(const Pose& pose0, Pose& pose1, RegisTf& first_tf) const;
    void init_base(RegisTf& first_tf);

    void get_transform(const RegisTf& last_res,
                       const Pose& last_pose,
                       RegisTf& curr_res,
                       Pose& curr_pose,
                       const Eigen::VectorXd& trans_vec,
                       const Eigen::VectorXd& scale_vec,
                       const Eigen::VectorXd& last_trans_vec,
                       const Eigen::VectorXd& last_scale_vec,
                       const bool& interval_2,
                       const Pose& base_pose);

    void get_transform(const RegisTf& res01, RegisTf& res12, RegisTf& res02, double* opt_vars, double* opt_sigs);

    static void PatternMatching_trans(const Eigen::VectorXd& last,
                                      const Eigen::VectorXd& curr,
                                      const double& init,
                                      const double& delta,
                                      const double& zp,
                                      double& lambda,
                                      double& sigma);

    static void PatternMatching_scale(const Eigen::VectorXd& last,
                                      const Eigen::VectorXd& curr,
                                      const double& init,
                                      double& lambda,
                                      double& sigma,
                                      std::string debug_name = "PMS");

    void PatternMatching_trans_scale(const Eigen::VectorXd& trans,
                                     const Eigen::VectorXd& scale,
                                     const double& init,
                                     double& lambda,
                                     double& sigma,
                                     std::string debug_name = "PMST",
                                     int key = 13);
    void PatternMatching_trans(const Eigen::VectorXd& last,
                               const Eigen::VectorXd& curr,
                               const double& init,
                               double& lambda,
                               double& sigma,
                               std::string debug_name = "PMT",
                               int key = 16);
    // void PatternMatching_scale(const Eigen::VectorXd &last, const Eigen::VectorXd &curr, const double &init, double &lambda, double &sigma);

  protected:
    cv::Mat im2_gray_;
    cv::Mat im2_logpolar_;
    cv::Mat im02_rotated_;

    double fx;
    double fy;

    double curr_scale_sigma;
    double curr_trans_sigma;
    double last_scale_sigma;
    double last_trans_sigma;
    double scale_sigma_02;
    double trans_sigma_02;
    double last_lambda_t;
    double curr_lambda_t;
    double last_lambda_s;
    double curr_lambda_s;
};

#endif /* IMAGE_REGISTRATION_H */
