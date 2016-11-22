/*
 * point_cloud2_gpu.h
 * Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
 */

#ifndef POINT_CLOUD2_GPU_H__
#define POINT_CLOUD2_GPU_H__

#include <stereo_image_proc/stereo_camera_model_gpu.h>
#include <stereo_image_proc/processor_gpu.h>
#include <stereo_image_proc/point_cloud2.h>
#include <cv_bridge/cv_bridge.h>


namespace stereo_image_proc {

  class GPUPointCloud2Nodelet : public PointCloud2Nodelet
  {
  protected:
    GPUStereoProcessor gpu_block_matcher_;
    GPUStereoCameraModel gpu_model_;

    void imageCb(const ImageConstPtr& l_image_msg,
                 const CameraInfoConstPtr& l_info_msg,
                 const CameraInfoConstPtr& r_info_msg,
                 const DisparityImageConstPtr& disp_msg);
  };

} // ns


#endif // POINT_CLOUD2_GPU_H__
