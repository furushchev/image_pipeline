/*
 * stereo_camera_model_gpu.h
 * Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
 */

#ifndef STEREO_CAMERA_MODEL_GPU_H__
#define STEREO_CAMERA_MODEL_GPU_H__


#include <image_geometry/stereo_camera_model.h>
#include <opencv2/cudastereo.hpp>

namespace stereo_image_proc {

  class GPUStereoCameraModel : public image_geometry::StereoCameraModel {

  public:
    void projectDisparityImageTo3dRaw(const cv::cuda::GpuMat& disparity,
                                      cv::cuda::GpuMat& point_cloud) const {
      assert( initialized() );
      cv::cuda::reprojectImageTo3D(disparity, point_cloud, Q_, 3);
    }

    void projectDisparityImageTo3d(const cv::Mat& disparity,
                                   const cv::Mat& point_cloud) const {
      cv::cuda::GpuMat d_disp, d_pc;
      d_disp.upload(disparity);
      projectDisparityImageTo3dRaw(d_disp, d_pc);
      d_pc.download(point_cloud);
    }
    void projectDisparityImageTo3d(const cv::Mat& disparity,
                                   const cv::Mat& point_cloud,
                                   bool handleMissingValues /* = unused */) const {
      projectDisparityImageTo3d(disparity, point_cloud);
    }
  };
} // ns

#endif // STEREO_CAMERA_MODEL_GPU_H__
