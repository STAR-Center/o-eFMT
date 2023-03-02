#include <image_transforms.h>
#include <math.h>
#include <opencv2/core/eigen.hpp>

ImageTransforms::ImageTransforms(int rows, int cols, int logPolarrows, int logPolarcols, int polarrows, int polarcols)
    : rows_(rows), cols_(cols), logPolarrows_(logPolarrows), logPolarcols_(logPolarcols), polarrows_(polarrows), polarcols_(polarcols),
      cv_xMap(logPolarrows, logPolarcols, CV_32FC1), cv_yMap(logPolarrows, logPolarcols, CV_32FC1),
      xMap(cv_xMap.ptr<float>(), cv_xMap.rows, cv_xMap.cols), yMap(cv_yMap.ptr<float>(), cv_yMap.rows, cv_yMap.cols),
      cv_xPolarMap(polarrows, polarcols, CV_32FC1), cv_yPolarMap(polarrows, polarcols, CV_32FC1),
      xPolarMap(cv_xPolarMap.ptr<float>(), cv_xPolarMap.rows, cv_xPolarMap.cols),
      yPolarMap(cv_yPolarMap.ptr<float>(), cv_yPolarMap.rows, cv_yPolarMap.cols) {
    // The xMap and yMap will be initialized with the same content address as the cv_xMat and cv_yMat.
    // So there is only construction of x/yMat in 'createLogPolarMap'.
    // logBase_ = std::exp(std::log(rows_ * 1.1 / 2.0) / std::max(rows_, cols_));
    createLogPolarMap();
    createPolarMap();
    border_mask_ = cv::Mat(rows_, cols_, CV_8UC1, cv::Scalar(255));

    Eigen::MatrixXd win = getApodizationWindow(rows_, cols_, (int)((0.12) * std::min(rows_, cols_)));
    cv::eigen2cv(win, appodizationWindow);
    appodizationWindow.convertTo(appodizationWindow, CV_32F);
}

ImageTransforms::~ImageTransforms() {}

void ImageTransforms::createLogPolarMap() {
    // logBase_ = e^{ \frac{ \log{rows_ * 1.1 / 2.0} }{ \max{(rows_, cols_)} } }
    logBase_ = std::exp(std::log(rows_ * 1.1 / 2.0) / std::max(rows_, cols_));
    Eigen::VectorXf scales(logPolarcols_);
    float ellipse_coefficient = (float)(rows_) / cols_;
    for (int i = 0; i < logPolarcols_; i++) {
        scales(i) = std::pow(logBase_, i);
    }
    Eigen::VectorXf ones_rows = Eigen::VectorXf::Ones(logPolarrows_);
    Eigen::MatrixXf scales_matrix = ones_rows * scales.transpose();  // identical rows, each col is logBase^i

    Eigen::VectorXf angles = Eigen::VectorXf::LinSpaced(logPolarrows_, 0.0, M_PI);
    angles *= -1.0;
    Eigen::MatrixXf angles_matrix = angles * Eigen::VectorXf::Ones(logPolarcols_).transpose();  // identical columns, each row is linspace 0-pi

    float centre[2] = {rows_ / 2.0f, cols_ / 2.0f};
    EigenRowMatrix cos = angles_matrix.array().cos() / ellipse_coefficient;
    EigenRowMatrix sin = angles_matrix.array().sin();
    xMap = scales_matrix.cwiseProduct(cos).array() + centre[1];  // row_major
    yMap = scales_matrix.cwiseProduct(sin).array() + centre[0];  // row_major
}

void ImageTransforms::createPolarMap() {
    float ellipse_coefficient = (float)(rows_) / cols_;
    Eigen::VectorXf rho = Eigen::VectorXf::LinSpaced(polarcols_, 0.0, std::min(polarrows_, polarcols_) * 0.75);
    Eigen::VectorXf ones_rows = Eigen::VectorXf::Ones(polarrows_);
    Eigen::MatrixXf rho_matrix = ones_rows * rho.transpose();  // identical rows, each col is logBase^i

    Eigen::VectorXf theta = Eigen::VectorXf::LinSpaced(polarcols_, 0.0, 2 * M_PI);
    Eigen::MatrixXf angles_matrix = theta * Eigen::VectorXf::Ones(polarcols_).transpose();  // identical columns, each row is linespace 0-pi

    float centre[2] = {rows_ / 2.0f, cols_ / 2.0f};
    EigenRowMatrix cos = angles_matrix.array().cos() / ellipse_coefficient;
    EigenRowMatrix sin = angles_matrix.array().sin();
    xPolarMap = rho_matrix.cwiseProduct(cos).array() + centre[1];  // row_major
    yPolarMap = rho_matrix.cwiseProduct(sin).array() + centre[0];  // row_major
}

void ImageTransforms::remapLogPolar(const cv::Mat& src, cv::Mat& dst) {
    // basically, dst(i,j) ~= src(cv_yMap(i,j), cv_xMap(i,j))
    // cv_yMap(i,j) = (power(logbase_, j) * sin(-i*180/pi)) + rows/2
    // cv_xMap(i,j) = (power(logbase_, j) * cos(-i*180/pi)) + cols/2
    cv::remap(src, dst, cv_xMap, cv_yMap, cv::INTER_CUBIC & cv::INTER_MAX, cv::BORDER_CONSTANT, cv::Scalar());
}

void ImageTransforms::remapPolar(const cv::Mat& src, cv::Mat& dst) {
    // basically, dst(i,j) ~= src(cv_yMap(i,j), cv_xMap(i,j))
    cv::remap(src, dst, cv_xPolarMap, cv_yPolarMap, cv::INTER_CUBIC & cv::INTER_MAX, cv::BORDER_CONSTANT, cv::Scalar());
}

void ImageTransforms::getTranslation(double row, double col, double& delta_row, double& delta_col, double& row_sigma) const {
    double rho = col / polarcols_ * (std::min(polarcols_, polarrows_) * 0.75);
    double phi = row / polarrows_ * 2.0 * M_PI;  // radian
    row_sigma = row_sigma / polarrows_ * 2.0 * M_PI;
    delta_row = rho * std::sin(phi);
    delta_col = rho * std::cos(phi);
}

void ImageTransforms::getScaleRotation(double row, double col, double& scale, double& rotation, double& row_sigma) const {
    rotation = row / polarrows_ * 180;
    row_sigma = row_sigma / polarrows_ * 180;
    scale = std::pow(logBase_, col);
    scale = 1.0 / scale;
}
#include <utility.h>
void ImageTransforms::rotateAndScale(const cv::Mat& src, cv::Mat& dst, double scale, double angle) {
    cv::Mat warped_image;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(src.cols / 2, src.rows / 2), angle, scale);
    //    debug_output("rotationMatrix", rotationMatrix);
    cv::warpAffine(src, dst, rotationMatrix, src.size());
    cv::warpAffine(border_mask_, rotated_mask_, rotationMatrix, src.size());
}

void ImageTransforms::translate(const cv::Mat& src, cv::Mat& dst, double xTrans, double yTrans) {
    cv::Mat warped_image;
    cv::Mat translateMatrix = (cv::Mat_<float>(2, 3) << 1.0, 0.0, xTrans, 0.0, 1.0, yTrans);
    cv::warpAffine(src, dst, translateMatrix, src.size());
    cv::warpAffine(rotated_mask_, rotated_translated_mask_, translateMatrix, src.size());
}

cv::Mat ImageTransforms::getBorderMask() {
    return rotated_translated_mask_;
}

void ImageTransforms::apodize(const cv::Mat& in, cv::Mat& out) {
    out = in.mul(appodizationWindow);
}

Eigen::VectorXd ImageTransforms::getHanningWindow(int size) {
    Eigen::VectorXd window(size);
    for (int i = 0; i < size; i++) {
        window(i) = 0.5 - 0.5 * std::cos((2 * M_PI * i) / (size - 1));
    }
    return window;
}

Eigen::MatrixXd ImageTransforms::getApodizationWindow(int rows, int cols, int radius) {
    Eigen::VectorXd hanning_window = getHanningWindow(radius * 2);

    Eigen::VectorXd row = Eigen::VectorXd::Ones(cols);
    row.segment(0, radius) = hanning_window.segment(0, radius);
    row.segment(cols - radius, radius) = hanning_window.segment(radius, radius);

    Eigen::VectorXd col = Eigen::VectorXd::Ones(rows);
    col.segment(0, radius) = hanning_window.segment(0, radius);
    col.segment(rows - radius, radius) = hanning_window.segment(radius, radius);

    return col * row.transpose();
}
