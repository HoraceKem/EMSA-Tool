/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __RHOANA_OPENCV_STITCHING_SEAM_FINDERS_HPP__
#define __RHOANA_OPENCV_STITCHING_SEAM_FINDERS_HPP__

#include <set>
//#include <opencv2/core/core.hpp>
//#include <opencv2/core/gpumat.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>

#include <opencv2/core.hpp>

namespace cv {
namespace detail {


class CV_EXPORTS RhoanaGraphCutSeamFinderBase
{
public:
    enum { COST_COLOR, COST_COLOR_GRAD };
};


class CV_EXPORTS RhoanaGraphCutSeamFinder : public RhoanaGraphCutSeamFinderBase, public SeamFinder
{
public:
    RhoanaGraphCutSeamFinder(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                       float bad_region_penalty = 1000.f);

    ~RhoanaGraphCutSeamFinder();

    void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
              std::vector<UMat> &masks);

private:
    // To avoid GCGraph dependency
    class Impl;
    Ptr<PairwiseSeamFinder> impl_;
};


#if 0
class CV_EXPORTS GraphCutSeamFinderGpu : public GraphCutSeamFinderBase, public PairwiseSeamFinder
{
public:
    GraphCutSeamFinderGpu(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                          float bad_region_penalty = 1000.f)
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
                          : cost_type_(cost_type),
                            terminal_cost_(terminal_cost),
                            bad_region_penalty_(bad_region_penalty)
#endif
    {
        (void)cost_type;
        (void)terminal_cost;
        (void)bad_region_penalty;
    }

    void find(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners,
              std::vector<cv::Mat> &masks);
    void findInPair(size_t first, size_t second, Rect roi);

private:
    void setGraphWeightsColor(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &mask1, const cv::Mat &mask2,
                              cv::Mat &terminals, cv::Mat &leftT, cv::Mat &rightT, cv::Mat &top, cv::Mat &bottom);
    void setGraphWeightsColorGrad(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &dx1, const cv::Mat &dx2,
                                  const cv::Mat &dy1, const cv::Mat &dy2, const cv::Mat &mask1, const cv::Mat &mask2,
                                  cv::Mat &terminals, cv::Mat &leftT, cv::Mat &rightT, cv::Mat &top, cv::Mat &bottom);
    std::vector<Mat> dx_, dy_;
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
#endif
};

#endif /* 0 */

} // namespace detail
} // namespace cv

#endif // __RHOANA_OPENCV_STITCHING_SEAM_FINDERS_HPP__
