/*
 * ImagesComposer - composes a panaroma similar to what composePanorama in OpenCV does.
 */

#include "ImagesComposer.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
//#include <opencv2/features2d.hpp>
#include <opencv2/stitching/warpers.hpp>
//#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
//#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include "detail/exposure_compensate.hpp"
//#include <opencv2/stitching/detail/seam_finders.hpp>
#include "detail/seam_finders.hpp"
//#include <opencv2/stitching/detail/blenders.hpp>
#include "detail/blenders.hpp"
//#include <opencv2/stitching/detail/camera.hpp>

using namespace cv;


#define CV_INSTRUMENT_REGION()
//#define LOGLN(x)
#define LOGLN(x) std::cout << x << std::endl;
#define ENABLE_LOG 1
#define LOG(x)


void convertToRGBImages(std::vector<UMat> &images)
{
    for (size_t i = 0; i < images.size(); i++)
    {
        assert(images[i].channels() == 1);
        cvtColor(images[i], images[i], CV_GRAY2RGB);
    }
}

int ImagesComposer::compose_panorama(
//int compose_panorama(
        InputArrayOfArrays in_warped_images, InputArrayOfArrays in_warped_masks, std::vector<Point>& warped_corners,
        float seam_scale,
        InputArrayOfArrays in_warped_seams_images, InputArrayOfArrays in_warped_seams_masks, std::vector<Point>& warped_seams_corners,
        OutputArray pano)
{

    /* Create the panorama's helpers */
    //double registr_resol_ = -1;
    //double seam_est_resol_ = 0.1;
    double compose_resol_ = 1;
    double work_scale_ = 1;
    //double seam_work_aspect_ = 1;
    ///Ptr<detail::SeamFinder> seam_finder_(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR));
    Ptr<detail::SeamFinder> seam_finder_(makePtr<detail::RhoanaGraphCutSeamFinder>(detail::RhoanaGraphCutSeamFinderBase::COST_COLOR));
    //Ptr<detail::SeamFinder> seam_finder_(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR_GRAD));
    ///Ptr<detail::ExposureCompensator> exposure_comp_(makePtr<detail::BlocksGainCompensator>());
    Ptr<detail::ExposureCompensator> exposure_comp_(makePtr<detail::RhoanaGainCompensator>());
    //Ptr<detail::ExposureCompensator> exposure_comp_(makePtr<detail::GainCompensator>());
    //Ptr<detail::ExposureCompensator> exposure_comp_(makePtr<detail::NoExposureCompensator>());
    ///Ptr<detail::Blender> blender_(makePtr<detail::MultiBandBlender>(false));
    Ptr<detail::RhoanaBlender> blender_(makePtr<detail::RhoanaMultiBandBlender>(false));
    //Ptr<detail::Blender> blender_(makePtr<detail::FeatherBlender>());
    //Ptr<detail::Blender> blender_(makePtr<detail::Blender>());

    std::vector<UMat> warped_images;
    std::vector<UMat> warped_masks;
    //std::vector<Point> warped_corners;
    std::vector<UMat> warped_seams_images;
    std::vector<UMat> warped_seams_masks;
    //std::vector<Point> warped_seams_corners;

    in_warped_images.getUMatVector(warped_images);
    ///convertToRGBImages(warped_images);
    in_warped_masks.getUMatVector(warped_masks);
    in_warped_seams_images.getUMatVector(warped_seams_images);
    ///convertToRGBImages(warped_seams_images);
    in_warped_seams_masks.getUMatVector(warped_seams_masks);

    std::vector<Size> warped_sizes(warped_images.size());

    /*
    // Copy the corners
    Mat in_warped_corners_mat = in_warped_corners.getMat();
    int n_corners = in_warped_corners_mat.checkVector(2);
    //int depth = points.depth();
    //CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    const Point* corners1 = in_warped_corners_mat.ptr<Point>();
    const Point* corners2 = in_warped_seams_corners.getMat().ptr<Point>();
    warped_corners.resize(warped_corners.size());
    warped_seams_corners.resize(warped_seams_corners.size());
    for (int i = 0; i < warped_corners.size(); i++)
    {
        warped_sizes[i] = warped_images[i].size();

        // populate the corners vectors
        warped_corners[i] = corners1[i];
        warped_seams_corners[i] = corners2[i];
    }
    */

    // Initialize the sizes
    for (size_t i = 0; i < warped_images.size(); i++)
    {
        warped_sizes[i] = warped_images[i].size();
    }

    LOGLN("Compensating exposure (pre-seams detection)")

    // Compensate exposure before finding seams
    exposure_comp_->feed(warped_seams_corners, warped_seams_images, warped_seams_masks);
    for (size_t i = 0; i < warped_seams_images.size(); ++i)
        exposure_comp_->apply(int(i), warped_seams_corners[i], warped_seams_images[i], warped_seams_masks[i]);

    LOGLN("Detecting seams")

    // Find seams
    std::vector<UMat> warped_seams_images_f(warped_seams_images.size());
    for (size_t i = 0; i < warped_seams_images.size(); ++i)
        warped_seams_images[i].convertTo(warped_seams_images_f[i], CV_32F);
    seam_finder_->find(warped_seams_images_f, warped_seams_corners, warped_seams_masks);

    LOGLN("Adi debug 2");

    // Release unused memory
    //seam_est_imgs_.clear();
    //images_warped.clear();
    //masks.clear();
    warped_seams_images_f.clear();
    warped_seams_corners.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    UMat img_warped, img_warped_s;
    UMat dilated_mask, seam_mask, mask, mask_warped;

    double compose_work_aspect = 1;
    bool is_blender_prepared = false;

    double compose_scale = 1;


    if (compose_resol_ > 0) // should be 1
        compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / warped_images[0].size().area()));

    // Compute relative scales
    //compose_seam_aspect = compose_scale / seam_scale_;
    compose_work_aspect = compose_scale / work_scale_;

    UMat full_img, img;
    for (size_t img_idx = 0; img_idx < warped_images.size(); ++img_idx)
    {
        LOGLN("Compositing image #" << img_idx + 1);
#if ENABLE_LOG
        int64 compositing_t = getTickCount();
#endif

#if ENABLE_LOG
        int64 pt = getTickCount();
#endif
        // Compensate exposure
        //std::cout << "Corners " << img_idx << ": " << warped_corners[img_idx] << std::endl;
        exposure_comp_->apply((int)img_idx, warped_corners[img_idx], warped_images[img_idx], warped_masks[img_idx]);
        LOGLN(" compensate exposure: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        warped_images[img_idx].convertTo(img_warped_s, CV_16S);

        // Make sure seam mask has proper size
        dilate(warped_seams_masks[img_idx], dilated_mask, UMat());
        resize(dilated_mask, seam_mask, warped_masks[img_idx].size());

        bitwise_and(seam_mask, warped_masks[img_idx], warped_masks[img_idx]);

        LOGLN(" other: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        if (!is_blender_prepared)
        {
            blender_->prepare(warped_corners, warped_sizes);
            is_blender_prepared = true;
        }

        LOGLN(" other2: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");

        LOGLN(" feed...");
#if ENABLE_LOG
        int64 feed_t = getTickCount();
#endif

        //std::cout << "warped_masks[0] type: " << warped_masks[0].type() << std::endl;
        // Blend the current image
        blender_->feed(img_warped_s, warped_masks[img_idx], warped_corners[img_idx]);
        LOGLN(" feed time: " << ((getTickCount() - feed_t) / getTickFrequency()) << " sec");
        LOGLN("Compositing ## time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");
    }

#if ENABLE_LOG
    int64 blend_t = getTickCount();
#endif
    UMat result, result_mask;
    blender_->blend(result, result_mask);
    LOGLN("blend time: " << ((getTickCount() - blend_t) / getTickFrequency()) << " sec");

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    warped_corners.clear();

    // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
    // so convert it to avoid user confusing
    //result.convertTo(pano, CV_8U);
    result.convertTo(pano, CV_8U);
    ///cvtColor(pano, pano, CV_RGB2GRAY);

    return 0;
}





