//
// Created by jiang on 2022/7/22.
//
#include "utility.h"

void Utility::print_transform(std::ofstream& out, Eigen::Matrix4d& transform) {
    for (int i = 0; i < transform.rows(); ++i) {
        for (int j = 0; j < transform.cols(); ++j) {
            out << transform(i, j) << " ";
        }
    }
    out << std::endl;
}

void Utility::print_transform(std::ofstream& out, Pose& pose) {
    out << pose.x << " " << pose.y << " " << pose.z << " " << pose.yaw << std::endl;
}

void Utility::print_transform(std::ofstream& out, RegisTf& res) {
    out << res.dx << " " << res.dy << " " << res.dz << " " << res.theta << " " << res.scale << " " << std::endl;
}

std::string Utility::get_time_string() {
    char buf[80];
    time_t now = time(0);
    tm* ltm = localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", ltm);
    return std::string(buf);
}

std::shared_ptr<spdlog::logger> Utility::set_logger(const std::string& log_name, const int level) {
    // Create a file rotating logger with 5mb size max and 3 rotated files
    auto max_size = 1024 * 1024 * 10;
    auto max_files = 10;

    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_name, max_size, max_files));
    auto combined_logger = std::make_shared<spdlog::logger>("efmt_logger", begin(sinks), end(sinks));
    spdlog::register_logger(combined_logger);

    if (level == Utility::LEVEL_ERROR) {
        spdlog::set_level(spdlog::level::err);  // Set global log level to error
    } else if (level == Utility::LEVEL_WARN) {
        spdlog::set_level(spdlog::level::warn);  // Set global log level to warning
    } else if (level == Utility::LEVEL_INFO) {
        spdlog::set_level(spdlog::level::info);  // Set global log level to info
    } else {
        spdlog::set_level(spdlog::level::debug);  // Set global log level to debug
    }
    spdlog::set_pattern("[%C/%m/%d %H:%M:%S.%e][thread %t][%s:%#,%!]%^[%=8l] : %v%$");
    return combined_logger;
}

void Utility::debug_output(std::string fname, Eigen::MatrixXd mat) {
    std::ofstream out;
    Eigen::IOFormat HeavyFmt(Eigen::FullPrecision);
    out.open("output/txt/" + fname + ".txt", std::ios::trunc | std::ios::out);
    out << mat.format(HeavyFmt);
    out.close();
}

void Utility::debug_output(std::string fname, cv::Mat mat) {
    std::ofstream out;
    out.open("output/txt/" + fname + ".txt", std::ios::trunc | std::ios::out);
    if (mat.type() == CV_64F) {
        for (int r = 0; r < mat.rows; r++) {
            auto data = mat.at<double>(r, 0);
            out << data;
            for (int c = 1; c < mat.cols; c++) {
                data = mat.at<double>(r, c);
                out << " " << data;
            }
            if (r != mat.rows - 1)
                out << std::endl;
        }
    } else if (mat.type() == CV_32F) {
        for (int r = 0; r < mat.rows; r++) {
            auto data = mat.at<float>(r, 0);
            out << data;
            for (int c = 1; c < mat.cols; c++) {
                data = mat.at<float>(r, c);
                out << " " << data;
            }
            if (r != mat.rows - 1)
                out << std::endl;
        }
    }

    out.close();
}

void Utility::load_imageName_and_timestamps(int startFrame,
                                            int endFrame,
                                            const std::string& data_root,
                                            std::vector<std::string>& imageFilenames,
                                            std::vector<std::string>& timeStamps) {
    std::string timestamps_path = data_root + "/timestamps.txt";
    std::string image_path = data_root + "/imageList.txt";
    std::ifstream in_image, in_time;
    in_image.open(image_path.c_str(), std::ios::in);
    in_time.open(timestamps_path.c_str(), std::ios::in);
    std::string image_name, timestamp;
    std::vector<std::string> names, times;
    int cnt = 0;
    while (!in_image.eof()) {
        in_image >> image_name;
        in_time >> timestamp;
        names.push_back(data_root + "/Images/" + image_name);
        times.push_back(timestamp);
        //        SPDLOG_LOGGER_INFO(Utility::logger, "{0} {1}", image_name, timestamp);
        cnt++;
    }
    if (endFrame == -1 || endFrame > cnt)
        endFrame = cnt - 1;
    imageFilenames.assign(names.begin() + startFrame, names.begin() + endFrame);
    timeStamps.assign(times.begin() + startFrame, times.begin() + endFrame);
    std::clog << "DONE:" << imageFilenames.size() << " / " << cnt << " images and timestamps are ready." << std::endl;
}

void Utility::load_intrinsics(const std::string& path2calib, cv::Mat& camera_intrinsic, cv::Mat& dist_coeffs, int& width, int& height) {
    cv::FileStorage fSettings(path2calib, cv::FileStorage::READ);
    camera_intrinsic = cv::Mat::eye(3, 3, CV_64F);
    camera_intrinsic.at<double>(0, 0) = (double)fSettings["Camera.fx"];
    camera_intrinsic.at<double>(1, 1) = (double)fSettings["Camera.fy"];
    camera_intrinsic.at<double>(0, 2) = (double)fSettings["Camera.cx"];
    camera_intrinsic.at<double>(1, 2) = (double)fSettings["Camera.cy"];
    dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    dist_coeffs.at<double>(0, 0) = (double)fSettings["Camera.k1"];
    dist_coeffs.at<double>(1, 0) = (double)fSettings["Camera.k2"];
    dist_coeffs.at<double>(2, 0) = (double)fSettings["Camera.p1"];
    dist_coeffs.at<double>(3, 0) = (double)fSettings["Camera.p2"];
    width = (int)fSettings["Camera.width"];
    height = (int)fSettings["Camera.height"];
    if (((int)camera_intrinsic.at<double>(0, 2) != width / 2) || ((int)camera_intrinsic.at<double>(1, 2) != height / 2)) {
        std::cerr << "The cx,cy(" << camera_intrinsic.at<double>(1, 2) << ", " << camera_intrinsic.at<double>(0, 2)
                  << ") is not on the center of the image(" << height << " * " << width << ")!!!" << std::endl;
    }
}

void Utility::get_image(std::string image_name, cv::Mat& out_image, const int& img_shape) {
    // image preprocess
    cv::Mat im = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
    int min_shape = im.cols < im.rows ? im.cols : im.rows;
    cv::Rect rect((im.cols - min_shape) / 2, (im.rows - min_shape) / 2, min_shape, min_shape);
    cv::Mat cut_img = im(rect);
    out_image.create(img_shape, img_shape, im.type());
    cv::resize(cut_img, out_image, out_image.size());
}

void Utility::check_img_shape(const int& width, const int& height, int& img_shape, cv::Mat& cameraMatrix) {
    //    cv::Mat im = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
    int min_shape = width < height ? width : height;
    if (min_shape > img_shape) {
        cameraMatrix.at<double>(0, 0) *= (img_shape * 1.0 / min_shape);
        cameraMatrix.at<double>(1, 1) *= (img_shape * 1.0 / min_shape);
    } else if (min_shape / 8 < img_shape) {
        std::cerr << "the shape of image is so small..." << std::endl;
        return;
    } else {
        img_shape = min_shape;
    }
    cameraMatrix.at<double>(0, 2) = 0;
    cameraMatrix.at<double>(1, 2) = 0;
}
