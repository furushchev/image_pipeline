/*
 * processor_gpu.h
 * Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
 */

#ifndef PROCESSOR_GPU_H__
#define PROCESSOR_GPU_H__

#include <stereo_image_proc/processor.h>
#include <stereo_image_proc/stereo_camera_model_gpu.h>
#include <opencv2/cudastereo.hpp>

namespace stereo_image_proc {

  class GPUStereoProcessor : public StereoProcessor {
  public:
    GPUStereoProcessor()
    {
      cu_block_matcher_ = cv::cuda::createStereoBM();
    }

    void cvPointCloudToPointCloud2(const cv::cuda::GpuMat& d_pc,
                                   const cv::Mat& color,
                                   const std::string encoding,
                                   sensor_msgs::PointCloud2& points) const;

    bool processDisparityRaw(const cv::cuda::GpuMat& left_rect,
                             const cv::cuda::GpuMat& right_rect,
                             cv::cuda::GpuMat& disparity) const;

    bool processDisparity(const cv::Mat& left_rect,
                          const cv::Mat& right_rect,
                          const GPUStereoCameraModel& model,
                          stereo_msgs::DisparityImage& disparity) const;

    bool processPoint2(const cv::Mat& disparity,
                       const cv::Mat& color, const std::string& encoding,
                       const GPUStereoCameraModel& model,
                       sensor_msgs::PointCloud2& points) const;

    bool processPoint2(const cv::Mat& left_rect,
                       const cv::Mat& right_rect,
                       const cv::Mat& color, const std::string& encoding,
                       const GPUStereoCameraModel& model,
                       sensor_msgs::PointCloud2& points) const;

  protected:
    mutable cv::Ptr<cv::cuda::StereoBM> cu_block_matcher_;
  };
} // ns

#endif // PROCESSOR_GPU_H__
