/*
 * processor_gpu.cpp
 * Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
 */

#include <stereo_image_proc/processor_gpu.h>

#include <ros/assert.h>
#include <sensor_msgs/image_encodings.h>
#include <cmath>
#include <limits>

namespace stereo_image_proc {

  bool GPUStereoProcessor::processDisparityRaw(const cv::cuda::GpuMat& left_rect,
                                               const cv::cuda::GpuMat& right_rect,
                                               cv::cuda::GpuMat& disparity) const
  {
    if (current_stereo_algorithm_ != BM) {
      ROS_ERROR("only supported BM with GPU");
      return false;
    }
    disparity = cv::cuda::GpuMat(left_rect.size(), CV_8U);
    cu_block_matcher_->compute(left_rect, right_rect, disparity);
    return true;
  }

  bool GPUStereoProcessor::processDisparity(const cv::Mat& left_rect,
                                            const cv::Mat& right_rect,
                                            const GPUStereoCameraModel& model,
                                            stereo_msgs::DisparityImage& disparity) const
  {
    static const int DPP = 16;
    static const double INV_DPP = 1.0 / DPP;

    cv::cuda::GpuMat d_left, d_right, d_disp;

    d_left.upload(left_rect);
    d_right.upload(right_rect);
    if(!processDisparityRaw(d_left, d_right, d_disp)) return false;
    d_disp.download(disparity16_);

    sensor_msgs::Image& dimage = disparity.image;
    dimage.height = disparity16_.rows;
    dimage.width = disparity16_.cols;
    dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    dimage.step = dimage.width * sizeof(float);
    dimage.data.resize(dimage.step * dimage.height);
    cv::Mat_<float> dmat(dimage.height, dimage.width, (float*)&dimage.data[0], dimage.step);
    // We convert from fixed-point to float disparity and also adjust for any x-offset between
    // the principal points: d = d_fp*inv_dpp - (cx_l - cx_r)
    disparity16_.convertTo(dmat, dmat.type(), INV_DPP, -(model.left().cx() - model.right().cx()));
    ROS_ASSERT(dmat.data == &dimage.data[0]);

    // Stereo parameters
    disparity.f = model.right().fx();
    disparity.T = model.baseline();

    /// @todo Window of (potentially) valid disparities

    // Disparity search range
    disparity.min_disparity = getMinDisparity();
    disparity.max_disparity = getMinDisparity() + getDisparityRange() - 1;
    disparity.delta_d = INV_DPP;

    // Adjust for any x-offset between the principal points: d' = d - (cx_l - cx_r)
    double cx_l = model.left().cx();
    double cx_r = model.right().cx();
    if (cx_l != cx_r) {
      cv::Mat_<float> disp_image(dimage.height, dimage.width,
                                 reinterpret_cast<float*>(&dimage.data[0]),
                                 dimage.step);
      cv::subtract(disp_image, cv::Scalar(cx_l - cx_r), disp_image);
    }

    return true;
  }

  inline bool isValidPoint(const cv::Vec3f& pt)
  {
    // Check both for disparities explicitly marked as invalid (where OpenCV maps pt.z to MISSING_Z)
    // and zero disparities (point mapped to infinity).
    return pt[2] != image_geometry::StereoCameraModel::MISSING_Z && !std::isinf(pt[2]);
  }


  void GPUStereoProcessor::cvPointCloudToPointCloud2(const cv::cuda::GpuMat& d_pc,
                                                     const cv::Mat& color,
                                                     const std::string encoding,
                                                     sensor_msgs::PointCloud2& points) const
  {
    cv::Mat pc;
    d_pc.download(pc);
    cv::Mat_<cv::Vec3f> dense_points = pc;

    // Fill in sparse point cloud message
    points.height = dense_points.rows;
    points.width  = dense_points.cols;
    points.fields.resize (4);
    points.fields[0].name = "x";
    points.fields[0].offset = 0;
    points.fields[0].count = 1;
    points.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    points.fields[1].name = "y";
    points.fields[1].offset = 4;
    points.fields[1].count = 1;
    points.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    points.fields[2].name = "z";
    points.fields[2].offset = 8;
    points.fields[2].count = 1;
    points.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    points.fields[3].name = "rgb";
    points.fields[3].offset = 12;
    points.fields[3].count = 1;
    points.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    //points.is_bigendian = false; ???
    points.point_step = 16;
    points.row_step = points.point_step * points.width;
    points.data.resize (points.row_step * points.height);
    points.is_dense = false; // there may be invalid points
 
    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    int i = 0;
    for (int32_t u = 0; u < dense_points.rows; ++u) {
      for (int32_t v = 0; v < dense_points.cols; ++v, ++i) {
        if (isValidPoint(dense_points(u,v))) {
          // x,y,z,rgba
          memcpy (&points.data[i * points.point_step + 0], &dense_points(u,v)[0], sizeof (float));
          memcpy (&points.data[i * points.point_step + 4], &dense_points(u,v)[1], sizeof (float));
          memcpy (&points.data[i * points.point_step + 8], &dense_points(u,v)[2], sizeof (float));
        }
        else {
          memcpy (&points.data[i * points.point_step + 0], &bad_point, sizeof (float));
          memcpy (&points.data[i * points.point_step + 4], &bad_point, sizeof (float));
          memcpy (&points.data[i * points.point_step + 8], &bad_point, sizeof (float));
        }
      }
    }

    // Fill in color
    namespace enc = sensor_msgs::image_encodings;
    i = 0;
    if (encoding == enc::MONO8) {
      for (int32_t u = 0; u < dense_points.rows; ++u) {
        for (int32_t v = 0; v < dense_points.cols; ++v, ++i) {
          if (isValidPoint(dense_points(u,v))) {
            uint8_t g = color.at<uint8_t>(u,v);
            int32_t rgb = (g << 16) | (g << 8) | g;
            memcpy (&points.data[i * points.point_step + 12], &rgb, sizeof (int32_t));
          }
          else {
            memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
          }
        }
      }
    }
    else if (encoding == enc::RGB8) {
      for (int32_t u = 0; u < dense_points.rows; ++u) {
        for (int32_t v = 0; v < dense_points.cols; ++v, ++i) {
          if (isValidPoint(dense_points(u,v))) {
            const cv::Vec3b& rgb = color.at<cv::Vec3b>(u,v);
            int32_t rgb_packed = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
            memcpy (&points.data[i * points.point_step + 12], &rgb_packed, sizeof (int32_t));
          }
          else {
            memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
          }
        }
      }
    }
    else if (encoding == enc::BGR8) {
      for (int32_t u = 0; u < dense_points.rows; ++u) {
        for (int32_t v = 0; v < dense_points.cols; ++v, ++i) {
          if (isValidPoint(dense_points(u,v))) {
            const cv::Vec3b& bgr = color.at<cv::Vec3b>(u,v);
            int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
            memcpy (&points.data[i * points.point_step + 12], &rgb_packed, sizeof (int32_t));
          }
          else {
            memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
          }
        }
      }
    }
    else {
      ROS_WARN("Could not fill color channel of the point cloud, unrecognized encoding '%s'", encoding.c_str());
    }
  }

  bool GPUStereoProcessor::processPoint2(const cv::Mat& disparity,
                                         const cv::Mat& color, const std::string& encoding,
                                         const GPUStereoCameraModel& model,
                                         sensor_msgs::PointCloud2& points) const
  {
    cv::Mat_<float> dispf = disparity;
    cv::cuda::GpuMat d_disp;
    d_disp.upload(dispf);

    cv::cuda::GpuMat d_pc;
    model.projectDisparityImageTo3dRaw(d_disp, d_pc);

    cvPointCloudToPointCloud2(d_pc, color, encoding, points);
    return true;
  }

  bool GPUStereoProcessor::processPoint2(const cv::Mat& left_rect,
                                         const cv::Mat& right_rect,
                                         const cv::Mat& color, const std::string& encoding,
                                         const GPUStereoCameraModel& model,
                                         sensor_msgs::PointCloud2& points) const
  {
    cv::cuda::GpuMat d_left, d_right, d_disp, d_pc;

    d_left.upload(left_rect);
    d_right.upload(right_rect);
    if(!processDisparityRaw(d_left, d_right, d_disp)) return false;

    // Adjust for any x-offset between the principal points: d' = d - (cx_l - cx_r)
    double cx_l = model.left().cx();
    double cx_r = model.right().cx();
    if (cx_l != cx_r) {
      ROS_WARN_ONCE("Parameter Cx of left/right camera is different. Process may be slow.");
      cv::Mat_<float> dimage;
      d_disp.download(dimage);
      cv::subtract(dimage, cv::Scalar(cx_l - cx_r), dimage);
      d_disp.upload(dimage);
    }

    model.projectDisparityImageTo3dRaw(d_disp, d_pc);
    cvPointCloudToPointCloud2(d_pc, color, encoding, points);
    return true;
  }

} // ns
