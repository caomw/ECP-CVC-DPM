#ifndef _LIB_PARTFILTER_H_
#define _LIB_PARTFILTER_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "LibLSVM.hpp"
#include "LibPyramid.h"

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

namespace liblsvm
{
// Transformation filter displacement from the block space
// to the space of pixels at the initial image
int convertPoints(int /*countLevel*/, int lambda,
                  int initialImageLevel,
                  CvPoint *points, int *levels,
                  CvPoint **partsDisplacement, int kPoints, int n,
                  int maxXBorder,
                  int maxYBorder);

// Elimination boxes that are outside the image boudaries
int clippingBoxes(int width, int height,
                  CvPoint *points, int kPoints);

// Computation of the root filter displacement and values of score function
int searchObject(const CvLSVMFeaturePyramid *H, const CvLSVMFilterObject **all_F,
                 int n, float b,
                 int maxXBorder,
                 int maxYBorder,
                 CvPoint **points, int **levels, int *kPoints, float *score,
                 CvPoint ***partsDisplacement);
// Computation right bottom corners coordinates of bounding boxes
static int estimateBoxes(CvPoint *points, int *levels, int kPoints, int sizeX, int sizeY, CvPoint **oppositePoints);

// Computation maximum filter size for each dimension
int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents, const int *kPartFilters, unsigned int *maxXBorder, unsigned int *maxYBorder);

// Perform non-maximum suppression algorithm (described in original paper)
// to remove "similar" bounding boxes
int nonMaximumSuppression(int numBoxes, const CvPoint *points, const CvPoint *oppositePoints, const float *score, float overlapThreshold, int *numBoxesOut, CvPoint **pointsOut, CvPoint **oppositePointsOut, float **scoreOut);

// Computation root filters displacement and values of score function
int searchObjectThresholdSomeComponents(const CvLSVMFeaturePyramid *H, const CvLSVMFilterObject **filters, int kComponents, const int *kPartFilters, const float *b, float scoreThreshold, CvPoint **points, CvPoint **oppPoints, float **score, int *kPoints, int numThreads);

// Release pyramid map
int freeFeaturePyramidObject (CvLSVMFeaturePyramid **obj);

// Computation border size for feature map
int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by);

// Computation the maximum of the score function
int maxFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                       const CvLSVMFeaturePyramid *H, float b,
                       int maxXBorder, int maxYBorder,
                       float *score,
                       CvPoint **points, int **levels, int *kPoints,
                       CvPoint ***partsDisplacement);

// Compute opposite point for filter box
int getOppositePoint(CvPoint point,
                     int sizeX, int sizeY,
                     float step, int degree,
                     CvPoint *oppositePoint);

// Computation of the root filter displacement and values of score function
int searchObjectThreshold(const CvLSVMFeaturePyramid *H,
                          const CvLSVMFilterObject **all_F, int n,
                          float b,
                          int maxXBorder, int maxYBorder,
                          float scoreThreshold,
                          CvPoint **points, int **levels, int *kPoints,
                          float **score, CvPoint ***partsDisplacement,
                          int numThreads);

// Computation the maximum of the score function at the level
int maxFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                 const CvLSVMFeaturePyramid *H,
                                 int level, float b,
                                 int maxXBorder, int maxYBorder,
                                 float *score, CvPoint **points,
                                 int *kPoints, CvPoint ***partsDisplacement);

// Function for convolution computation
int convolution(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *map, float *f);

// Computation score function that exceed threshold
int thresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                             const CvLSVMFeaturePyramid *H,
                             float b,
                             int maxXBorder, int maxYBorder,
                             float scoreThreshold,
                             float **score,
                             CvPoint **points, int **levels, int *kPoints,
                             CvPoint ***partsDisplacement);

// Computation score function at the level that exceed threshold
int thresholdFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                       const CvLSVMFeaturePyramid *H,
                                       int level, float b,
                                       int maxXBorder, int maxYBorder,
                                       float scoreThreshold,
                                       float **score, CvPoint **points, int *kPoints,
                                       CvPoint ***partsDisplacement);

static CvLSVMFeatureMap* featureMapBorderPartFilter(CvLSVMFeatureMap *map,
                                       int maxXBorder, int maxYBorder);

// Computation objective function D according the original paper
int filterDispositionLevel(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *pyramid,
                           float **scoreFi,
                           int **pointsX, int **pointsY);

}

#endif