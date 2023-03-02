#include "optimizer.h"
#include "utility.h"
#include <fstream>
#include <image_registration.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>

typedef std::string string;
int img_shape = 512;
bool base_version = false;
bool display_imgs = false;
int frame_idx = 1;
std::shared_ptr<spdlog::logger> Utility::logger;

int main(int argc, char** argv) {
    string data_root = string(argv[1]);
    string output_root = string(argv[2]);
    int st_frame = argc >=4?std::stoi(string(argv[3])):0;
    int ed_frame = argc >=5?std::stoi(string(argv[4])):-1;
    // display_imgs = argc >=6?bool(argv[5]):false;
    // base_version = argc >=7?bool(argv[6]):false;
    string path2imgs = data_root + "/Images";
    string path2calib = data_root + "/cam_param.yaml";
    string path2Tf_base = output_root + "/tf_base.txt";
    string path2Tf_opt = output_root + "/tf_opt.txt";

    std::string log_name = Utility::get_time_string() + "-log.txt";
    Utility::logger = Utility::set_logger("log/" + log_name, Utility::LEVEL_DEBUG);

    // load images and timestamps
    std::vector<string> imgNames;
    std::vector<string> timeStamps;
    std::cout << "test here..........." << std::endl;
    Utility::load_imageName_and_timestamps(st_frame, ed_frame, data_root, imgNames, timeStamps);
    //    Utility::load_imageName_and_timestamps(80, 110, data_root, imgNames, timeStamps);
    SPDLOG_LOGGER_INFO(Utility::logger, "Load {0:4d} images and timestamps from {1}.", imgNames.size(), data_root);
    std::cout << "Load {0:4d} images and timestamps from {1}" << std::endl;

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    int width, height;
    Utility::load_intrinsics(path2calib, cameraMatrix, distCoeffs, width, height);
    Utility::check_img_shape(width, height, img_shape, cameraMatrix);
    SPDLOG_LOGGER_INFO(Utility::logger, "The focal length is fx={0}, fy={1}.", cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1));

    // prepare the key data saver
    std::ofstream out_tf_base(path2Tf_base, std::ios::trunc | std::ios::out);
    std::ofstream out_tf_opt(path2Tf_opt, std::ios::out | std::ios::trunc);

    // store the poses and registration results
    RegisTf res01(50), res12(50), res02(50);
    Utility::print_transform(out_tf_base, res01);
    Utility::print_transform(out_tf_opt, res01);

    //    ****************************************************
    //     ******* the first VIO and initialize the lambda
    cv::Mat im0, im1, im2;
    Utility::get_image(imgNames[0], im0, img_shape);
    Utility::get_image(imgNames[1], im1, img_shape);

    // define the register and optimizer
    ImageRegistrationOpt image_registration(im0, im1, cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1));
    Optimizer optimizer(cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1));

    cv::Mat registered_image;
    image_registration.registerImage01(registered_image, res01, display_imgs, base_version);

    // initialize: set the scale base and base vectors
    image_registration.init_base(res01);
    SPDLOG_LOGGER_INFO(Utility::logger, "frame 0-1: delta_x = {0}, delta_y = {1}, theta = {2}, scale = {3}.", res01.dx, res01.dy, res01.theta,
                       res01.scale);
    Utility::print_transform(out_tf_base, res01);
    Utility::print_transform(out_tf_opt, res01);
    //    pose0 = pose1;

    // ********* VO from the second one ************
    double sum_time = 0;
    for (frame_idx = 2; frame_idx < imgNames.size(); ++frame_idx) {
        optimizer.set_frame_idx(frame_idx);
        Utility::get_image(imgNames[frame_idx], im2, img_shape);
        SPDLOG_LOGGER_INFO(Utility::logger, "==== processing frame {0}... ====", frame_idx);
        if (display_imgs) {
            cv::imshow("im0", image_registration.getPreviousImage());
            cv::imshow("im1", image_registration.getCurrentImage());
            cv::Mat overlay_image;
            cv::addWeighted(image_registration.getCurrentImage(), 0.5, registered_image, 0.5, 0.0, overlay_image);
            cv::imshow("overlay_image", overlay_image);
            cv::imwrite(output_root + "/overlay_image_" + std::to_string(frame_idx - 2) + "_" + std::to_string(frame_idx - 1) + ".tiff",
                        overlay_image);
        }
        // 1) get the parameters and two vectors by our method
        if (!base_version) {
            // the first frame will be registed with the third one
            image_registration.registerImage02(im2, registered_image, res02, display_imgs, base_version);
            if (display_imgs) {
                cv::imshow("im2", image_registration.getLastImage());
                cv::Mat overlay_image;
                cv::addWeighted(image_registration.getLastImage(), 0.5, registered_image, 0.5, 0.0, overlay_image);
                cv::imshow("overlay_image02", overlay_image);
                cv::imwrite(output_root + "/overlay_image_" + std::to_string(frame_idx - 2) + " " + std::to_string(frame_idx) + ".tiff",
                            overlay_image);
                cv::waitKey(500);
            }
        }
        image_registration.next();
        clock_t st_time, ed_time;
        st_time = clock();
        image_registration.registerImage01(registered_image, res12, display_imgs, base_version);
        ed_time = clock();
        sum_time += (double)(ed_time - st_time)/CLOCKS_PER_SEC;

        SPDLOG_LOGGER_INFO(Utility::logger, "VVVV before optimizing {0}... VVVV", frame_idx);
        SPDLOG_LOGGER_INFO(Utility::logger, "frame {4}-{5}: delta_x = {0}, delta_y = {1}, theta = {2}, scale = {3}.", res01.dx, res01.dy, res01.theta,
                           res01.scale, frame_idx - 2, frame_idx - 1);
        SPDLOG_LOGGER_INFO(Utility::logger, "frame {4}-{5}: delta_x = {0}, delta_y = {1}, theta = {2}, scale = {3}.", res12.dx, res12.dy, res12.theta,
                           res12.scale, frame_idx - 1, frame_idx);
        SPDLOG_LOGGER_INFO(Utility::logger, "frame {4}-{5}: delta_x = {0}, delta_y = {1}, theta = {2}, scale = {3}.", res02.dx, res02.dy, res02.theta,
                           res02.scale, frame_idx - 2, frame_idx);

        // 2) PM for the vectors to find lambda and restore the transform with the parameters and lambda
        // Done: get optimize variable from the registration results
        double opt_vars[10];
        double opt_sigs[10];
        image_registration.get_transform(res01, res12, res02, opt_vars, opt_sigs);
        SPDLOG_LOGGER_INFO(Utility::logger,
                           "frame {0}: PMT01_12={1}, PMT01_02={2}, PMS01_12={3}, PMS01_12={4}, PMST12={5}, PMST02={6}, phi12={7}, phi02={8}, "
                           "theta12={9}, theta02={10}",
                           frame_idx, opt_vars[0], opt_vars[1], opt_vars[2], opt_vars[3], opt_vars[4], opt_vars[5], opt_vars[6], opt_vars[7],
                           opt_vars[8], opt_vars[9]);
        SPDLOG_LOGGER_INFO(Utility::logger,
                           "opt_sigs : PMT01_12={1}, PMT01_02={2}, PMS01_12={3}, PMS01_12={4}, PMST12={5}, PMST02={6}, phi12={7}, phi02={8}, "
                           "theta12={9}, theta02={10}",
                           frame_idx, opt_sigs[0], opt_sigs[1], opt_sigs[2], opt_sigs[3], opt_sigs[4], opt_sigs[5], opt_sigs[6], opt_sigs[7],
                           opt_sigs[8], opt_sigs[9]);
        Utility::print_transform(out_tf_base, res12);

        // 3) optimize for the local closure, this part should set in optimize part
        // optimize: T_1^2T_0^1 - T_0^2
        // optimizer.check_before_optimize(res01, res12, res02, opt_vars);
        SPDLOG_LOGGER_INFO(Utility::logger,
                           "frame {0}: PMT01_12={1}, PMT01_02={2}, PMS01_12={3}, PMS01_12={4}, PMST12={5}, PMST02={6}, phi12={7}, phi02={8}, "
                           "theta12={9}, theta02={10}",
                           frame_idx, opt_vars[0], opt_vars[1], opt_vars[2], opt_vars[3], opt_vars[4], opt_vars[5], opt_vars[6], opt_vars[7],
                           opt_vars[8], opt_vars[9]);
        SPDLOG_LOGGER_INFO(Utility::logger,
                           "opt_sigs : PMT01_12={1}, PMT01_02={2}, PMS01_12={3}, PMS01_12={4}, PMST12={5}, PMST02={6}, phi12={7}, phi02={8}, "
                           "theta12={9}, theta02={10}",
                           frame_idx, opt_sigs[0], opt_sigs[1], opt_sigs[2], opt_sigs[3], opt_sigs[4], opt_sigs[5], opt_sigs[6], opt_sigs[7],
                           opt_sigs[8], opt_sigs[9]);
        optimizer.optimize(res01, res12, res02, opt_vars, opt_sigs);
        // optimizer.update_windows_after_optimize(res01, res12);
        SPDLOG_LOGGER_INFO(Utility::logger, "---- after optimizing {0}... ----", frame_idx);
        SPDLOG_LOGGER_INFO(Utility::logger, "frame {4}-{5}: delta_x = {0}, delta_y = {1}, theta = {2}, scale = {3}.", res12.dx, res12.dy, res12.theta,
                           res12.scale, frame_idx - 1, frame_idx);
        SPDLOG_LOGGER_INFO(Utility::logger, "frame {4}-{5}: delta_x = {0}, delta_y = {1}, theta = {2}, scale = {3}.", res02.dx, res02.dy, res02.theta,
                           res02.scale, frame_idx - 2, frame_idx);
        Utility::print_transform(out_tf_opt, res12);

        // update for next step
        res01 = res12;
    }
    // Utility::print_transform(out_tf_opt, res12);


    // Done!
    out_tf_base.close();
    out_tf_opt.close();
    SPDLOG_LOGGER_INFO(Utility::logger, "Everything is done!, and each registration spend {0}s.", sum_time/(imgNames.size()-2));
    std::cout << "Everything is done!, and each registration spend " << sum_time/(imgNames.size()-2) << " s." << std::endl;
    spdlog::drop_all();

    return 0;
}
