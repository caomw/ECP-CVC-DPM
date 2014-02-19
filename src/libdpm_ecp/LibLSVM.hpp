#ifndef _LIB_LSVM_HPP_
#define _LIB_LSVM_HPP_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <map>
#include <deque>
#include <stdio.h>
#include <iostream>
#include "string.h"
#include "float.h"

#define LATENT_SVM_OK 0
#define LATENT_SVM_MEM_NULL 2
#define DISTANCE_TRANSFORM_OK 1
#define DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR -1
#define DISTANCE_TRANSFORM_ERROR -2
#define DISTANCE_TRANSFORM_EQUAL_POINTS -3
#define LATENT_SVM_GET_FEATURE_PYRAMID_FAILED -4
#define LATENT_SVM_SEARCH_OBJECT_FAILED -5
#define LATENT_SVM_FAILED_SUPERPOSITION -6
#define FILTER_OUT_OF_BOUNDARIES -7
#define LATENT_SVM_TBB_SCHEDULE_CREATION_FAILED -8
#define LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT -9
#define FFT_OK 2
#define FFT_ERROR -10
#define LSVM_PARSER_FILE_NOT_FOUND -11

#define MODEL    1
#define P        2
#define COMP     3
#define SCORE    4
#define RFILTER  100
#define PFILTERs 101
#define PFILTER  200
#define SIZEX    150
#define SIZEY    151
#define WEIGHTS  152
#define TAGV     300
#define Vx       350
#define Vy       351
#define TAGD     400
#define Dx       451
#define Dy       452
#define Dxx      453
#define Dyy      454
#define BTAG     500

#define STEP_END 1000

#define EMODEL    (STEP_END + MODEL)
#define EP        (STEP_END + P)
#define ECOMP     (STEP_END + COMP)
#define ESCORE    (STEP_END + SCORE)
#define ERFILTER  (STEP_END + RFILTER)
#define EPFILTERs (STEP_END + PFILTERs)
#define EPFILTER  (STEP_END + PFILTER)
#define ESIZEX    (STEP_END + SIZEX)
#define ESIZEY    (STEP_END + SIZEY)
#define EWEIGHTS  (STEP_END + WEIGHTS)
#define ETAGV     (STEP_END + TAGV)
#define EVx       (STEP_END + Vx)
#define EVy       (STEP_END + Vy)
#define ETAGD     (STEP_END + TAGD)
#define EDx       (STEP_END + Dx)
#define EDy       (STEP_END + Dy)
#define EDxx      (STEP_END + Dxx)
#define EDyy      (STEP_END + Dyy)
#define EBTAG     (STEP_END + BTAG)

//#define FFT_CONV true

#define PI    CV_PI

#define EPS 0.000001

#define F_MAX FLT_MAX
#define F_MIN -FLT_MAX

// The number of elements in bin
// The number of sectors in gradient histogram building
#define NUM_SECTOR 9

// The number of levels in image resize procedure
// We need Lambda levels to resize image twice
#define LAMBDA 10

// Block size. Used in feature pyramid building procedure
#define SIDE_LENGTH 8

#define VAL_OF_TRUNCATE 0.2f

using namespace std;
using namespace cv;

namespace liblsvm
{
// Filter postion
typedef struct CvLSVMFilterPosition
{
    int x;
    int y;
    int l;
}_CvLSVMFilterPosition;

// Object part filter
typedef struct CvLSVMFilterObject
{
    CvLSVMFilterPosition V;
    float fineFunction[4];
    int sizeX;
    int sizeY;
    int numFeatures;
    float *H;
}_CvLSVMFilterObject;

// Detector
typedef struct CvLatentSvmDetector
{
    int num_filters;
    int num_components;
    int* num_part_filters;
    CvLSVMFilterObject** filters;
    float* b;
    float score_threshold;
}_CvLatentSvmDetector;

// Struct for filter detection result
typedef struct CvObjectDetection
{
    CvRect rect;
    float score;
}_CvObjectDetection;

// feature Map
typedef struct CvLSVMFeatureMap
{
    int sizeX;
    int sizeY;
    int numFeatures;
    float *map;
}_CvLSVMFeatureMap;

// feature Pyramid
typedef struct CvLSVMFeaturePyramid
{
    int numLevels;
    CvLSVMFeatureMap **pyramid;
}_CvLSVMFeaturePyramid;

typedef struct CvLSVMFilterDisposition
{
    float *score;
    int *x;
    int *y;
}_CvLSVMFilterDisposition;

typedef struct CvLSVMFftImage
{
    int numFeatures;
    int dimX;
    int dimY;
    float **channels;
}_CvLSVMFftImage;

typedef struct CvLSVMFftPyramid
{
    int numLevels;
    CvLSVMFftImage **pyramid;
}_CvLSVMFftPyramid;

// Model file loading utility funciton
string extractModelName( const string& filename );

// Resize image
IplImage* resize_opencv(IplImage* img, float scale);

// sort the float array
void sort(int n, const float* x, int* indices);

// Decision of two dimensional problem generalized distance transform
// on the regular grid at all points
//      min{d2(y' - y) + d4(y' - y)(y' - y) +
//          min(d1(x' - x) + d3(x' - x)(x' - x) + f(x',y'))} (on x', y')
int DistanceTransformTwoDimensionalProblem( 
                                            const float *f,
                                            const int n, 
                                            const int m,
                                            const float coeff[4],
                                            float *distanceTransform,
                                            int *pointsX, int *pointsY );

// Decision of one dimensional problem generalized distance transform
// on the regular grid at all points
//      min (a(y' - y) + b(y' - y)(y' - y) + f(y')) (on y')
int DistanceTransformOneDimensionalProblem( const float *f, const int n,
                                            const float a, const float b,
                                            float *distanceTransform,
                                            int *points );

// Computation the point of intersection functions
// (parabolas on the variable y)
//      a(y - q1) + b(q1 - y)(q1 - y) + f[q1]
//      a(y - q2) + b(q2 - y)(q2 - y) + f[q2]
int GetPointOfIntersection( const float *f,
                            const float a, const float b,
                           int q1, int q2, float *point);

// Getting transposed matrix
void Transpose(float *a, int n, int m);

// Getting transposed matrix
static void Transpose_int(int *a, int n, int m);

// Computation next cycle element
int GetNextCycleElement(int k, int n, int q);

// Transpose cycle elements
void TransposeCycleElements(float *a, int *cycle, int cycle_len);

// Transpose cycle elements
static void TransposeCycleElements_int(int *a, int *cycle, int cycle_len);

int loadModel(
              const char *modelPath,
              CvLSVMFilterObject ***filters,
              int *kFilters,
              int *kComponents,
              int **kPartFilters,
              float **b,
              float *scoreThreshold);
int LSVMparser(const char * filename, CvLSVMFilterObject *** model, int *last, int *max, int **comp, float **b, int *count, float * score);
int isMODEL (char *str);
int isP     (char *str);
int isSCORE (char *str);
int isCOMP  (char *str);
int isRFILTER  (char *str);
int isPFILTERs (char *str);
int isPFILTER  (char *str);
int isSIZEX    (char *str);
int isSIZEY    (char *str);
int isWEIGHTS  (char *str);
int isV        (char *str);
int isVx       (char *str);
int isVy       (char *str);
int isD        (char *str);
int isDx       (char *str);
int isDy       (char *str);
int isDxx      (char *str);
int isDyy      (char *str);
int isB        (char *str);
int getTeg     (char *str);
void addFilter(CvLSVMFilterObject *** model, int *last, int *max);
void parserRFilter  (FILE * xmlf, int p, CvLSVMFilterObject * model, float *b);
void parserV  (FILE * xmlf, int /*p*/, CvLSVMFilterObject * model);
void parserD  (FILE * xmlf, int /*p*/, CvLSVMFilterObject * model);
void parserPFilter  (FILE * xmlf, int p, int /*N_path*/, CvLSVMFilterObject * model);
void parserPFilterS (FILE * xmlf, int p, CvLSVMFilterObject *** model, int *last, int *max);
void parserComp (FILE * xmlf, int p, int *N_comp, CvLSVMFilterObject *** model, float *b, int *last, int *max);
void parserModel(FILE * xmlf, CvLSVMFilterObject *** model, int *last, int *max, int **comp, float **b, int *count, float * score);
}
#endif