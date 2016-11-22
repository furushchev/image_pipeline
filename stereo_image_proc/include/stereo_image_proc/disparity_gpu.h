/*
 * disparity_gpu.h
 * Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
 */

#ifndef DISPARITY_GPU_H__
#define DISPARITY_GPU_H__

#include <stereo_image_proc/processor_gpu.h>
#include <stereo_image_proc/disparity.h>
#include <stereo_image_proc/stereo_camera_model_gpu.h>


namespace stereo_image_proc {

  class GPUDisparityNodelet : public DisparityNodelet
  {
  protected:
    GPUStereoProcessor gpu_block_matcher_;
    GPUStereoCameraModel gpu_model_;

    // override callback
    void imageCb(const ImageConstPtr& l_image_msg, const CameraInfoConstPtr& l_info_msg,
                 const ImageConstPtr& r_image_msg, const CameraInfoConstPtr& r_info_msg);
  };
} // ns

#endif // DISPARITY_GPU_H__
