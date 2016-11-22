/*
 * disparity_gpu.cpp
 * Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
 */

#include <stereo_image_proc/disparity_gpu.h>

namespace enc = sensor_msgs::image_encodings;

namespace stereo_image_proc {

  void GPUDisparityNodelet::imageCb(const ImageConstPtr& l_image_msg,
                                    const CameraInfoConstPtr& l_info_msg,
                                    const ImageConstPtr& r_image_msg,
                                    const CameraInfoConstPtr& r_info_msg)
  {  // Update the camera model
    gpu_model_.fromCameraInfo(l_info_msg, r_info_msg);

    // Allocate new disparity image message
    DisparityImagePtr disp_msg = boost::make_shared<DisparityImage>();
    disp_msg->header         = l_info_msg->header;
    disp_msg->image.header   = l_info_msg->header;

    // Compute window of (potentially) valid disparities
    int border   = gpu_block_matcher_.getCorrelationWindowSize() / 2;
    int left   = gpu_block_matcher_.getDisparityRange() + gpu_block_matcher_.getMinDisparity() + border - 1;
    int wtf = (gpu_block_matcher_.getMinDisparity() >= 0) ? border + gpu_block_matcher_.getMinDisparity() : std::max(border, -gpu_block_matcher_.getMinDisparity());
    int right  = disp_msg->image.width - 1 - wtf;
    int top    = border;
    int bottom = disp_msg->image.height - 1 - border;
    disp_msg->valid_window.x_offset = left;
    disp_msg->valid_window.y_offset = top;
    disp_msg->valid_window.width    = right - left;
    disp_msg->valid_window.height   = bottom - top;

    // Create cv::Mat views onto all buffers
    const cv::Mat_<uint8_t> l_image = cv_bridge::toCvShare(l_image_msg, enc::MONO8)->image;
    const cv::Mat_<uint8_t> r_image = cv_bridge::toCvShare(r_image_msg, enc::MONO8)->image;

    // Perform block matching to find the disparities
    gpu_block_matcher_.processDisparity(l_image, r_image, gpu_model_, *disp_msg);

    pub_disparity_.publish(disp_msg);
  }

} // ns

// Register nodelet
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(stereo_image_proc::GPUDisparityNodelet,nodelet::Nodelet)
