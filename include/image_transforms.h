#ifndef IMAGE_TRANSFORMS_H_
#define IMAGE_TRANSFORMS_H_

#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> EigenMap;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenRowMatrix;

class ImageTransforms {
  public:
    ImageTransforms(int rows, int cols, int logPolarrows, int logPolarcols, int polarrows, int polarcols);

    virtual ~ImageTransforms();

    /*
     * creat a table for looking up the corresponding logpolar coordinates
     */
    void createLogPolarMap();

    void createPolarMap();

    void remapLogPolar(const cv::Mat& src, cv::Mat& dst);

    void remapPolar(const cv::Mat& src, cv::Mat& dst);

    /*
     * Convert a row/col on IDFT of CPS of log-polar images into a scale and rotation
     */
    void getScaleRotation(double row, double col, double& scale, double& rotation, double& row_sigma) const;

    void getTranslation(double row, double col, double& x, double& y, double& row_sigma) const;

    /*
     * Rotate and scale image
     */
    void rotateAndScale(const cv::Mat& src, cv::Mat& dst, double scale, double angle);

    /*
     * translate image
     */
    void translate(const cv::Mat& src, cv::Mat& dst, double xTrans, double yTrans);

    /*
     * smooth borders
     */
    void apodize(const cv::Mat& in, cv::Mat& out);

    cv::Mat getBorderMask();
    double getLogbase() {
        return logBase_;
    };

  protected:
    static Eigen::VectorXd getHanningWindow(int size);

    Eigen::MatrixXd getApodizationWindow(int rows, int cols, int radius);

  protected:
    int rows_;
    int cols_;
    int logPolarrows_;
    int logPolarcols_;
    int polarrows_;
    int polarcols_;
    cv::Mat cv_xMap;  // save the x axis coordinates, same with the column.
    cv::Mat cv_yMap;  // save the y_axis coordinates, same with the row.
    EigenMap xMap;    // x/yMap and cv_x/yMap have the same data address(row_major)
    EigenMap yMap;
    cv::Mat cv_xPolarMap;
    cv::Mat cv_yPolarMap;
    EigenMap xPolarMap;
    EigenMap yPolarMap;

    double logBase_;
    cv::Mat border_mask_;
    cv::Mat rotated_mask_;
    cv::Mat rotated_translated_mask_;
    cv::Mat appodizationWindow;
};

#endif /* IMAGE_TRANSFORMS_H */
