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

//#include "precomp.hpp"
#include "exposure_compensate.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/detail/util.hpp>

#define CV_INSTRUMENT_REGION()
#define LOGLN(x) std::cout << x << std::endl;
#define ENABLE_LOG 1
#define LOG(x)
#include <iostream>

using namespace std;

namespace cv {
namespace detail {

void RhoanaGainCompensator::feed(const vector<Point> &corners, const vector<UMat> &images,
                           const vector<pair<UMat,uchar> > &masks)
{
    LOGLN("Exposure compensation...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());
    Mat_<int> N(num_images, num_images); N.setTo(0);
    Mat_<double> I(num_images, num_images); I.setTo(0);

    //Rect dst_roi = resultRoi(corners, images);
    Mat subimg1, subimg2;
    Mat_<uchar> submask1, submask2, intersect;

    for (int i = 0; i < num_images; ++i)
    {
        //std::cout << "Here1.1" << std::endl;
        for (int j = i; j < num_images; ++j)
        {
            Rect roi;
            if (overlapRoi(corners[i], corners[j], images[i].size(), images[j].size(), roi))
            {
                subimg1 = images[i](Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                subimg2 = images[j](Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);

                submask1 = masks[i].first(Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                submask2 = masks[j].first(Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);
                intersect = (submask1 == masks[i].second) & (submask2 == masks[j].second);

                N(i, j) = N(j, i) = std::max(1, countNonZero(intersect));

                double Isum1 = 0, Isum2 = 0;
                for (int y = 0; y < roi.height; ++y)
                {
                    ///const Point3_<uchar>* r1 = subimg1.ptr<Point3_<uchar> >(y);
                    ///const Point3_<uchar>* r2 = subimg2.ptr<Point3_<uchar> >(y);
                    const uchar* r1 = subimg1.ptr<uchar>(y);
                    const uchar* r2 = subimg2.ptr<uchar>(y);
                    for (int x = 0; x < roi.width; ++x)
                    {
                        if (intersect(y, x))
                        {
                            ///Isum1 += sqrt(static_cast<double>(sqr(r1[x].x) + sqr(r1[x].y) + sqr(r1[x].z)));
                            ///Isum2 += sqrt(static_cast<double>(sqr(r2[x].x) + sqr(r2[x].y) + sqr(r2[x].z)));
                            Isum1 += sqrt(static_cast<double>(sqr(r1[x])));
                            Isum2 += sqrt(static_cast<double>(sqr(r2[x])));
                        }
                    }
                }
                I(i, j) = Isum1 / N(i, j);
                I(j, i) = Isum2 / N(i, j);
            }
        }
        //std::cout << "Here1.2" << std::endl;
    }

    double alpha = 0.01;
    double beta = 100;

    Mat_<double> A(num_images, num_images); A.setTo(0);
    Mat_<double> b(num_images, 1); b.setTo(0);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            b(i, 0) += beta * N(i, j);
            A(i, i) += beta * N(i, j);
            if (j == i) continue;
            A(i, i) += 2 * alpha * I(i, j) * I(i, j) * N(i, j);
            A(i, j) -= 2 * alpha * I(i, j) * I(j, i) * N(i, j);
        }
    }

    solve(A, b, gains_);

    LOGLN("Exposure compensation, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


//void RhoanaGainCompensator::apply(int index, Point /*corner*/, Mat &image, const Mat &/*mask*/)
void RhoanaGainCompensator::apply(int index, Point /*corner*/, InputOutputArray image, InputArray /*mask*/)
{
    //image *= gains_(index, 0);
    CV_INSTRUMENT_REGION()

    multiply(image, gains_(index, 0), image);
}


vector<double> RhoanaGainCompensator::gains() const
{
    vector<double> gains_vec(gains_.rows);
    for (int i = 0; i < gains_.rows; ++i)
        gains_vec[i] = gains_(i, 0);
    return gains_vec;
}


void RhoanaBlocksGainCompensator::feed(const vector<Point> &corners, const vector<UMat> &images,
                                     const vector<pair<UMat,uchar> > &masks)
{
    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());

    vector<Size> bl_per_imgs(num_images);
    vector<Point> block_corners;
    vector<UMat> block_images;
    vector<pair<UMat,uchar> > block_masks;

    // Construct blocks for gain compensator
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        Size bl_per_img((images[img_idx].cols + bl_width_ - 1) / bl_width_,
                        (images[img_idx].rows + bl_height_ - 1) / bl_height_);
        int bl_width = (images[img_idx].cols + bl_per_img.width - 1) / bl_per_img.width;
        int bl_height = (images[img_idx].rows + bl_per_img.height - 1) / bl_per_img.height;
        bl_per_imgs[img_idx] = bl_per_img;
        for (int by = 0; by < bl_per_img.height; ++by)
        {
            for (int bx = 0; bx < bl_per_img.width; ++bx)
            {
                Point bl_tl(bx * bl_width, by * bl_height);
                Point bl_br(min(bl_tl.x + bl_width, images[img_idx].cols),
                            min(bl_tl.y + bl_height, images[img_idx].rows));

                block_corners.push_back(corners[img_idx] + bl_tl);
                block_images.push_back(images[img_idx](Rect(bl_tl, bl_br)));
                block_masks.push_back(make_pair(masks[img_idx].first(Rect(bl_tl, bl_br)),
                                                masks[img_idx].second));
            }
        }
    }

    RhoanaGainCompensator compensator;
    compensator.feed(block_corners, block_images, block_masks);
    vector<double> gains = compensator.gains();
    gain_maps_.resize(num_images);

    Mat_<float> ker(1, 3);
    ker(0,0) = 0.25; ker(0,1) = 0.5; ker(0,2) = 0.25;


    int bl_idx = 0;
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        Size bl_per_img = bl_per_imgs[img_idx];
        gain_maps_[img_idx].create(bl_per_img, CV_32F);

        {
            Mat_<float> gain_map = gain_maps_[img_idx].getMat(ACCESS_WRITE);
            for (int by = 0; by < bl_per_img.height; ++by)
                for (int bx = 0; bx < bl_per_img.width; ++bx, ++bl_idx)
                    gain_map(by, bx) = static_cast<float>(gains[bl_idx]);
        }

        sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
        sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
    }

}


void RhoanaBlocksGainCompensator::apply(int index, Point /*corner*/, InputOutputArray _image, InputArray /*mask*/)
{
    CV_INSTRUMENT_REGION()

    ///CV_Assert(image.type() == CV_8UC3);
    CV_Assert(_image.type() == CV_8UC1);

    UMat u_gain_map;
    if (gain_maps_[index].size() == _image.size())
        u_gain_map = gain_maps_[index];
    else
        resize(gain_maps_[index], u_gain_map, _image.size(), 0, 0, INTER_LINEAR);

    Mat_<float> gain_map = u_gain_map.getMat(ACCESS_READ);
    Mat image = _image.getMat();
    for (int y = 0; y < image.rows; ++y)
    {
        const float* gain_row = gain_map.ptr<float>(y);
        ///Point3_<uchar>* row = image.ptr<Point3_<uchar> >(y);
        uchar* row = image.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            ///row[x].x = saturate_cast<uchar>(row[x].x * gain_row[x]);
            ///row[x].y = saturate_cast<uchar>(row[x].y * gain_row[x]);
            ///row[x].z = saturate_cast<uchar>(row[x].z * gain_row[x]);
            row[x] = saturate_cast<uchar>(row[x] * gain_row[x]);
        }
    }
}

} // namespace detail
} // namespace cv
