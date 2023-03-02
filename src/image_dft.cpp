#include <math.h>

#include "image_dft.h"
#include "utility.h"

ImageDFT::ImageDFT(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    in_.resize(rows_, cols_);
    out_.resize(rows_, cols_);
    iin_.resize(rows_, cols_);
    iout_.resize(rows_, cols_);
    forward_plan_ = fftw_plan_dft_2d(rows_, cols_, const_cast<fftw_complex*>(reinterpret_cast<fftw_complex*>(in_.data())),
                                     const_cast<fftw_complex*>(reinterpret_cast<fftw_complex*>(out_.data())), FFTW_FORWARD, FFTW_MEASURE);
    backward_plan_ = fftw_plan_dft_2d(rows_, cols_, const_cast<fftw_complex*>(reinterpret_cast<fftw_complex*>(iin_.data())),
                                      const_cast<fftw_complex*>(reinterpret_cast<fftw_complex*>(iout_.data())), FFTW_BACKWARD, FFTW_MEASURE);
}

ImageDFT::~ImageDFT() {
    fftw_destroy_plan(forward_plan_);
    fftw_destroy_plan(backward_plan_);
}

ComplexMatrix ImageDFT::fft(const cv::Mat& im) {
    ComplexMatrix in(im.rows, im.cols);
    // TODO: find a better way to copy the data over
    for (int i = 0; i < im.rows; i++) {
        for (int j = 0; j < im.cols; j++) {
            in(i, j) = std::complex<double>(im.at<float>(i, j), 0.0);
        }
    }
    return fft(in);
}

ComplexMatrix ImageDFT::fft(const ComplexMatrix& in) {
    in_ = in;
    fftw_execute(forward_plan_);
    return out_;
}

ComplexMatrix ImageDFT::ifft(const ComplexMatrix& in) {
    iin_ = in;
    fftw_execute(backward_plan_);
    // output is scaled by size
    iout_ = iout_ / (iout_.size());
    return iout_;
}

//
// TODO: use templating so just one function is needed
// TODO: check if row/col size is even or odd
ComplexMatrix ImageDFT::fftShift(const ComplexMatrix& in) {
    ComplexMatrix out(in.rows(), in.cols());
    int block_rows = in.rows() / 2;
    int block_cols = in.cols() / 2;
    // swap first and third quadrant
    out.block(0, 0, block_rows, block_cols) = in.block(block_rows, block_cols, block_rows, block_cols);
    out.block(block_rows, block_cols, block_rows, block_cols) = in.block(0, 0, block_rows, block_cols);
    // swap second and fourth quadrant
    out.block(block_rows, 0, block_rows, block_cols) = in.block(0, block_cols, block_rows, block_cols);
    out.block(0, block_cols, block_rows, block_cols) = in.block(block_rows, 0, block_rows, block_cols);
    return out;
}

Eigen::MatrixXd ImageDFT::fftShift(const Eigen::MatrixXd& in) {
    Eigen::MatrixXd out(in.rows(), in.cols());
    int block_rows = in.rows() / 2;
    int block_cols = in.cols() / 2;
    // swap first and third quadrant
    out.block(0, 0, block_rows, block_cols) = in.block(block_rows, block_cols, block_rows, block_cols);
    out.block(block_rows, block_cols, block_rows, block_cols) = in.block(0, 0, block_rows, block_cols);
    // swap second and fourth quadrant
    out.block(block_rows, 0, block_rows, block_cols) = in.block(0, block_cols, block_rows, block_cols);
    out.block(0, block_cols, block_rows, block_cols) = in.block(block_rows, 0, block_rows, block_cols);
    return out;
}

Eigen::VectorXd ImageDFT::fftShift(const Eigen::VectorXd& in) {
    // [1, 2, 3, 4, 5] -> [4, 5, 1, 2, 3]
    // [1, 2, 3, 4, 5, 6] -> [4, 5, 6, 1, 2, 3]
    Eigen::VectorXd out(in.size());
    int block_size = in.size() / 2;
    int first_block_size = block_size;
    int second_start_point = block_size;
    if (in.size() % 2 != 0) {
        first_block_size++;
        second_start_point++;
    }
    // swap first half and second half
    out.segment(block_size, first_block_size) = in.segment(0, first_block_size);
    out.segment(0, block_size) = in.segment(first_block_size, block_size);
    return out;
}

Eigen::VectorXd ImageDFT::ifftShift(const Eigen::VectorXd& in) {
    // [4, 5, 1, 2, 3] -> [1, 2, 3, 4, 5]
    // [4, 5, 6, 1, 2, 3] -> [1, 2, 3, 4, 5, 6]
    Eigen::VectorXd out(in.size());
    int block_size = in.size() / 2;
    int second_block_size = block_size;
    int first_start_point = block_size;
    if (in.size() % 2 != 0) {
        second_block_size++;
        first_start_point++;
    }
    // swap first half and second half
    out.segment(first_start_point, block_size) = in.segment(0, block_size);
    out.segment(0, second_block_size) = in.segment(block_size, second_block_size);
    return out;
}

ComplexMatrix ImageDFT::crossPowerSpectrum(const ComplexMatrix& f1, const ComplexMatrix& f2) {
    // (f1 x f2*) / (abs(f1) x abs(f2))
    double eps = 1e-15;  // add eps in case denominator is zero
    Eigen::MatrixXd denom = f1.cwiseAbs().cwiseProduct(f2.cwiseAbs()).array() + eps;
    ComplexMatrix out = (f1.cwiseProduct(f2.conjugate())).cwiseQuotient(denom);
    return out;
}

Eigen::MatrixXd ImageDFT::getPSD(const cv::Mat& im0, const cv::Mat& im1) {
    ComplexMatrix m0 = fft(im0);
    ComplexMatrix m1 = fft(im1);
    ComplexMatrix cross_power = crossPowerSpectrum(m0, m1);
    ComplexMatrix inversefft = ifft(cross_power);
    ComplexMatrix shifted_cps = fftShift(inversefft);
    Eigen::MatrixXd abs = shifted_cps.cwiseAbs();
    return abs;
}

void ImageDFT::getPosAndVector(const Eigen::MatrixXd& abs,
                               double& row,
                               double& col,
                               double& row_sigma,
                               Eigen::VectorXd& energy_vector,
                               bool base_version) {
    int approx_row, approx_col;
    double max_peak;
    max_peak = abs.maxCoeff(&approx_row, &approx_col);  // argmax
                                             //    for scale-rot, the position is around center(row/2, col/2)
                                             //    for translation, the position is close to the first column. The col(related to rho) is small.

    int radius = 2;  // This is for higher precision
    Eigen::MatrixXd subArray = neighbourhood(abs, approx_row, approx_col, radius);
    getCentreOfMass(subArray, row, col);

    row = approx_row + (row - radius);
    col = approx_col + (col - radius);

    if (base_version)
        return;

    // TODO: optimize the method finding the proper row between row and max_row
    Eigen::MatrixXd abs_power = abs.array().cube();
    if (col > 20){
        int num = 5;
        abs_power.block(0,0,abs.rows(),num) = Eigen::MatrixXd::Zero(abs_power.rows(),num);
    }
    abs_power *= std::pow(10, -1 * int(log10(abs_power.maxCoeff())));  // in case the value is too small
    Eigen::VectorXd row_sums = abs_power.rowwise().sum();
    int max_row, max_col;  // the row with max sum, the max column in this row
    row_sums.maxCoeff(&max_row);
    abs_power.row(max_row).maxCoeff(&max_col);

    SPDLOG_LOGGER_DEBUG(Utility::logger, "the global peak is at approx_row = {0}, approx_col = {1}.", approx_row, approx_col);
    SPDLOG_LOGGER_DEBUG(Utility::logger, "row = {0}, col = {1}.", row, col);
    SPDLOG_LOGGER_DEBUG(Utility::logger, "the maximum of the max-sum row is at max_row = {0}, max_col = {1}.", max_row, max_col);

    // all the value in the first column is same and large, so the proportion will be large when rho = 0.
    // TODO: the parameter 0.25 may need to be adjusted
    if (approx_col == 0) {
        Eigen::VectorXd col_sums = abs_power.colwise().sum();
        if (col_sums(0) / col_sums.sum() > 0.25) {
            max_row = row;
        }
    }
    if (std::abs(approx_row - max_row) > radius && std::abs(approx_row - max_row) < row_sums.size() - radius) {
        if (std::abs(max_row - approx_row) <= 5) {
            radius = std::abs(max_row - approx_row);
        }else if(std::abs(max_row - approx_row) <= 10){
            max_row = row;
        }else{
            SPDLOG_LOGGER_WARN(Utility::logger, "The approx row {0} and max_row {1} is very different!", approx_row, max_row);
        }
        // max_row = approx_row;
        // max_col = approx_col;
        // save the abs_power
        Utility::debug_output("abs_power_" + std::to_string(frame_idx - 1) + "_" + std::to_string(frame_idx), abs_power);
        Utility::debug_output("abs_" + std::to_string(frame_idx - 1) + "_" + std::to_string(frame_idx), abs);
        Utility::debug_output("rowsum_" + std::to_string(frame_idx - 1) + "_" + std::to_string(frame_idx), row_sums);
    }

    // Eigen::VectorXd temp_ev = Eigen::VectorXd::Zero(abs_power.cols());
    // energy_vector = Eigen::VectorXd::Zero(abs_power.cols());
    // double maxv_j = 0;  // store the maximum in each column
    // double max_sum = 0;
    // for (int j = 0; j < temp_ev.size(); j++){
    //     maxv_j = 0;
    //     for (int i = 0; i < 2*radius+1; i++){
    //         int rid = (max_row - radius + i + row_sums.size()) % row_sums.size();
    //         maxv_j = maxv_j < abs_power(rid, j) ? abs_power(rid, j) : maxv_j;
    //     }
    //     if (maxv_j > 0.2 * max_peak){
    //         temp_ev[j] = maxv_j;
    //     }
    // }
    // int min_j, max_j;
    // double min_ev = temp_ev.minCoeff(&min_j);
    // double max_ev = temp_ev.maxCoeff(&max_j);
    // int small_j = min_j < max_j?min_j:max_j;
    // int large_j = min_j > max_j?min_j:max_j;
    // int peak_num = 5;
    // int step = (large_j - small_j) / peak_num;
    // for (int i = small_j; i <= large_j; i+=step){
    //     energy_vector(i) = temp_ev[small_j] + (i-small_j) * (temp_ev(large_j) - temp_ev(small_j)) / (large_j-small_j);
    // }

    // sub_row_sums over bound
    Eigen::VectorXd sub_row_sums = Eigen::VectorXd::Zero(radius * 2 + 1);
    // ..., sz-r, sz-r+1, ..., sz-1, 0, 1, ..., r-1, r, ...
    for (int i = 0; i < 2 * radius + 1; ++i) {
        sub_row_sums(i) = row_sums((max_row - radius + i + row_sums.size()) % row_sums.size());
    }

    Eigen::VectorXd idxes(2 * radius + 1);
    for (int i = 0; i < 2 * radius + 1; ++i) {
        idxes(i) = i - radius;
    }
    double Ex = idxes.dot(sub_row_sums) / sub_row_sums.sum();
    double avg_sigma = 0;
    int round_Ex = int(Ex);
    int cnt = 0;
    for (int i = -1 * radius; i <= radius; ++i) {
        if (round_Ex != i) {
            avg_sigma += std::sqrt(std::abs(((i - Ex) * (i - Ex) - (round_Ex - Ex) * (round_Ex - Ex)) /
                                            (-2 * std::log(sub_row_sums(i + radius) / sub_row_sums(radius + round_Ex)))));
            cnt++;
        }
    }
    row_sigma = avg_sigma / cnt + 0.01;
    //    double Exx = idxes.cwiseProduct(idxes).dot(sub_row_sums) / sub_row_sums.sum();
    //    row_sigma = std::sqrt(Exx - Ex * Ex);
    row = fmod(max_row + Ex + abs_power.rows(), abs_power.rows());

    Eigen::MatrixXd sub_block = Eigen::MatrixXd::Zero(radius * 2 + 1, abs_power.cols());
    for (int i = 0; i < 2 * radius + 1; ++i) {
        sub_block.row(i) = abs_power.row((max_row - radius + i + abs_power.rows()) % abs_power.rows());
    }

    // TODO: calculate the uncertainty of rho(column)
    // Done: utilize the sigma to optimize the energy_vector
    Eigen::VectorXd weight = (((idxes.array() - Ex) / row_sigma).pow(2) * -0.5).exp().inverse();
    energy_vector = (weight.transpose() * sub_block).transpose() / (2 * radius + 1);
    // energy_vector /= (energy_vector[col]);

    // int kernel_sz = 5;
    // Eigen::VectorXd gauss_kenel(kernel_sz);
    // gauss_kenel << 17, 66, 107, 66, 17;
    // gauss_kenel /= 273;
    // // Eigen::VectorXd gauss_kenel(kernel_sz);
    // // gauss_kenel << 17, 38, 49, 38, 17;
    // // gauss_kenel /= 159;
    // // Eigen::VectorXd gauss_kenel(kernel_sz);
    // // gauss_kenel << 1, 2, 1;
    // // gauss_kenel /= 4;
    // Eigen::VectorXd ev = energy_vector;
    // for (int i = kernel_sz/2; i < ev.size()-kernel_sz/2; i++){
    //     ev[i] = energy_vector.segment(i-kernel_sz/2, kernel_sz).dot(gauss_kenel);
    // }
    // energy_vector = ev;
    // energy_vector.maxCoeff(&col);
    SPDLOG_LOGGER_DEBUG(Utility::logger, "mu, sigma, row, col = {0}, {1}, {2}, {3}.", Ex, row_sigma, row, col);
}

void ImageDFT::phaseCorrelate(const cv::Mat& im0, const cv::Mat& im1, double& row, double& col) {
    Eigen::MatrixXd abs = getPSD(im0, im1);
    /**
     * This is the key part of the register algorithm. How to find the row and col and optimize the value.
     * rotation & scale: find the column and return the definite theta(row), scale_energy_vector)
     * translation: find the direction and return the translation_energy_vector, theta with uncertainty)
     */
    int approx_row, approx_col;
    abs.maxCoeff(&approx_row, &approx_col);  // argmax

    int radius = 2;  // This is for higher precision
    Eigen::MatrixXd subArray = neighbourhood(abs, approx_row, approx_col, radius);
    getCentreOfMass(subArray, row, col);

    row = approx_row + (row - radius);
    col = approx_col + (col - radius);
    row = row - (abs.rows() / 2);
    col = col - (abs.cols() / 2);
}

void ImageDFT::getCentreOfMass(const Eigen::MatrixXd& f1, double& row, double& col) {
    Eigen::VectorXd cols = Eigen::VectorXd::LinSpaced(f1.cols(), 0, f1.cols() - 1);
    Eigen::VectorXd rows = Eigen::VectorXd::LinSpaced(f1.rows(), 0, f1.rows() - 1);

    Eigen::MatrixXd colMat = Eigen::VectorXd::Ones(f1.rows()) * cols.transpose();
    Eigen::MatrixXd rowMat = rows * Eigen::VectorXd::Ones(f1.cols()).transpose();
    //    std::cout << colMat << std::endl;
    //    std::cout << rowMat << std::endl;

    double f1sum = f1.sum();
    col = colMat.cwiseProduct(f1).sum() / f1sum;
    row = rowMat.cwiseProduct(f1).sum() / f1sum;
}

Eigen::MatrixXd ImageDFT::getHighPassFilter() {
    Eigen::VectorXd yy = Eigen::VectorXd::LinSpaced(rows_, -M_PI / 2.0, M_PI / 2.0);
    Eigen::VectorXd yy_vec = Eigen::VectorXd::Ones(cols_);
    Eigen::MatrixXd yy_matrix = yy * yy_vec.transpose();  // identical cols, each row is linspace

    Eigen::VectorXd xx = Eigen::VectorXd::LinSpaced(cols_, -M_PI / 2.0, M_PI / 2.0);
    Eigen::VectorXd xx_vec = Eigen::VectorXd::Ones(rows_);
    Eigen::MatrixXd xx_matrix = xx_vec * xx.transpose();

    Eigen::MatrixXd filter = (yy_matrix.cwiseProduct(yy_matrix) + xx_matrix.cwiseProduct(xx_matrix)).cwiseSqrt().array().cos();
    filter = filter.cwiseProduct(filter);
    filter = -filter;
    filter = filter.array() + 1.0;
    return filter;
}

Eigen::MatrixXd ImageDFT::neighbourhood(const Eigen::MatrixXd& f1, int centre_row, int centre_col, int radius) {
    int size = 1 + radius * 2;
    Eigen::MatrixXd subArray(size, size);
    int row_start = centre_row - radius;
    int row_end = centre_row + radius;
    int col_start = centre_col - radius;
    int col_end = centre_col + radius;
    // neighbourhood falls within original size
    if (row_start > 0 && row_end < f1.rows() && col_start > 0 && col_end < f1.cols()) {
        subArray = f1.block(row_start, col_start, size, size);
    }
    // need to wrap around
    else {
        for (int i = 0; i < size; i++) {
            int ii = (i + row_start + f1.rows()) % f1.rows();
            for (int j = 0; j < size; j++) {
                int jj = (j + col_start + f1.cols()) % f1.cols();
                subArray(i, j) = f1(ii, jj);
            }
        }
    }
    return subArray;
}
