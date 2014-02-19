#ifndef _LIB_PYRAMID_H_
#define _LIB_PYRAMID_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "LibLSVM.hpp"

namespace liblsvm 
{

// Creation feature pyramid with nullable border
CvLSVMFeaturePyramid* createFeaturePyramidWithBorder( IplImage *image, int maxXBorder, int maxYBorder);

// Release pyramid map
int freeFeaturePyramidObject (CvLSVMFeaturePyramid **obj);

// Computation border size for feature map
int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by);

int freeFeatureMapObject (CvLSVMFeatureMap **obj);

int allocFeatureMapObject(CvLSVMFeatureMap **obj, const int sizeX,
                          const int sizeY, const int numFeatures);

// Getting feature pyramid
int getFeaturePyramid(IplImage * image, CvLSVMFeaturePyramid **maps);

// Addition nullable border to the feature map
int addNullableBorder(CvLSVMFeatureMap *map, int bx, int by);

int allocFeaturePyramidObject(CvLSVMFeaturePyramid **obj,
                              const int numLevels);

CvLSVMFftPyramid * createFftFeaturePyramid( CvLSVMFeaturePyramid **maps );

int allocFftFeaturePyramidObject( CvLSVMFftPyramid ** fft_obj, const int numLevels );
int allocFFTImage(CvLSVMFftImage **image, int numFeatures, int dimX, int dimY);
int freeFftFeaturePyramidObject( CvLSVMFftPyramid ** fft_obj );

static int getPathOfFeaturePyramid(IplImage * image,
                            float step, int numStep, int startIndex,
                            int sideLength, CvLSVMFeaturePyramid **maps);

// Getting feature map for the selected subimage
int getFeatureMaps(const IplImage* image, const int k, CvLSVMFeatureMap **map);

// Feature map Normalization and Truncation
int normalizeAndTruncate(CvLSVMFeatureMap *map, const float alfa);

// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
int PCAFeatureMaps(CvLSVMFeatureMap *map);

}// liblsvm

#endif