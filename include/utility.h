//
// Created by jiang on 2022/7/22.
//

#ifndef IMREG_FMT_UTILITY_H
#define IMREG_FMT_UTILITY_H

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <vector>

struct RegisTf {
    double zp;
    double zl;
    double dx;
    double dy;
    double dz;
    double scale;
    double theta;
    double phi;
    double sigma_phi;
    double sigma_z;
    double sigma_scale;
    double sigma_theta;

    RegisTf() : zp(0), zl(0), dx(0), dy(0), dz(0), scale(1), theta(0), phi(0), sigma_phi(0), sigma_z(0), sigma_scale(0), sigma_theta(0){};
    RegisTf(double depth) : zp(depth), zl(0), dx(0), dy(0), dz(0), scale(1), theta(0), phi(0), sigma_phi(0), sigma_z(0), sigma_scale(0), sigma_theta(0){};

    RegisTf(RegisTf const& tf)
        : zp(tf.zp), zl(tf.zl), dx(tf.dx), dy(tf.dy), dz(tf.dz), scale(tf.scale), theta(tf.theta), phi(tf.phi), sigma_phi(tf.sigma_phi), sigma_z(tf.sigma_z),
          sigma_scale(tf.sigma_scale), sigma_theta(tf.sigma_theta){};
};

struct Pose {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;

    Pose() : x(0.0), y(0.0), z(70), roll(0), pitch(0), yaw(0){};
};

class Utility {
  public:
    enum ID_Enum { LEVEL_DEBUG = 0, LEVEL_INFO = 1, LEVEL_WARN = 2, LEVEL_ERROR = 3 };

    static std::shared_ptr<spdlog::logger> logger;  // set logger before using it

    static void print_transform(std::ofstream& out, Eigen::Matrix4d& transform);

    static void print_transform(std::ofstream& out, Pose& pose);

    static void print_transform(std::ofstream& out, RegisTf& res);

    static std::string get_time_string();

    static std::shared_ptr<spdlog::logger> set_logger(const std::string& log_name, const int level = LEVEL_DEBUG);

    //    static void debug_output(std::string fname, Eigen::MatrixXf mat);
    static void debug_output(std::string fname, Eigen::MatrixXd mat);

    static void debug_output(std::string fname, cv::Mat mat);

    static void get_image(std::string image_name, cv::Mat& out_image, const int& img_shape);
    static void check_img_shape(const int& width, const int& height, int& img_shape, cv::Mat& cameraMatrix);

    static void load_imageName_and_timestamps(int startFrame,
                                              int endFrame,
                                              const std::string& data_root,
                                              std::vector<std::string>& imageFilenames,
                                              std::vector<std::string>& timeStamps);

    static void load_intrinsics(const std::string& path2calib, cv::Mat& camera_intrinsic, cv::Mat& dist_coeffs, int& width, int& height);
};

#endif  // IMREG_FMT_UTILITY_H
