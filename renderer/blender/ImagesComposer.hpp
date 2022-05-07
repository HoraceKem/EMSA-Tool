#ifndef _IMAGES_COMPOSER_HPP_
#define _IMAGES_COMPOSER_HPP_

#include <opencv2/core/core.hpp>

using namespace cv;

class ImagesComposer
{
  public:
    static int compose_panorama(
        InputArrayOfArrays in_warped_images, InputArrayOfArrays in_warped_masks, std::vector<Point>& in_warped_corners,
        float seam_scale,
        InputArrayOfArrays in_warped_seams_images, InputArrayOfArrays in_warped_seams_masks, std::vector<Point>& in_warped_seams_corners,
        OutputArray pano);
    
};

#endif /* _IMAGES_COMPOSER_HPP_ */
