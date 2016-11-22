/*
 * point_cloud2_gpu.cpp
 * Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
 */


#include <stereo_image_proc/point_cloud2_gpu.h>

namespace enc = sensor_msgs::image_encodings;

namespace stereo_image_proc {
  void GPUPointCloud2Nodelet::imageCb(const ImageConstPtr& l_image_msg,
                                      const CameraInfoConstPtr& l_info_msg,
                                      const CameraInfoConstPtr& r_info_msg,
                                      const DisparityImageConstPtr& disp_msg)
  {
    gpu_model_.fromCameraInfo(l_info_msg, r_info_msg);
    const cv::Mat color = cv_bridge::toCvShare(l_image_msg)->image;
    const sensor_msgs::Image &dimg = disp_msg->image;
    const cv::Mat_<float> disparity(dimg.height, dimg.width, (float*)&dimg.data[0], dimg.step);

    sensor_msgs::PointCloud2 pc_msg;
    gpu_block_matcher_.processPoint2(disparity, color,
                                     l_image_msg->encoding,
                                     gpu_model_, pc_msg);
    pub_points2_.publish(pc_msg);
  }
} // ns

// Register nodelet
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(stereo_image_proc::GPUPointCloud2Nodelet,nodelet::Nodelet)


