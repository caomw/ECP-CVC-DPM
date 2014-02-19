#include "LibPartFilter.h"

namespace liblsvm
{
/*
// Transformation filter displacement from the block space
// to the space of pixels at the initial image
//
// API
// int convertPoints(int countLevel, CvPoint *points, int *levels,
                  CvPoint **partsDisplacement, int kPoints, int n);
// INPUT
// countLevel        - the number of levels in the feature pyramid
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// partsDisplacement - displacement of part filters (in the block space)
// kPoints           - number of root filter positions
// n                 - number of part filters
// initialImageLevel - level that contains features for initial image
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// points            - the set of root filter positions (in the space of pixels)
// partsDisplacement - displacement of part filters (in the space of pixels)
// RESULT
// Error status
*/
int convertPoints(int /*countLevel*/, int lambda,
                  int initialImageLevel,
                  CvPoint *points, int *levels,
                  CvPoint **partsDisplacement, int kPoints, int n,
                  int maxXBorder,
                  int maxYBorder)
{
    int i, j, bx, by;
    float step, scale;
    step = powf( 2.0f, 1.0f / ((float)lambda) );

    computeBorderSize(maxXBorder, maxYBorder, &bx, &by);

    for (i = 0; i < kPoints; i++)
    {
        // scaling factor for root filter
        scale = SIDE_LENGTH * powf(step, (float)(levels[i] - initialImageLevel));
        points[i].x = (int)((points[i].x - bx + 1) * scale);
        points[i].y = (int)((points[i].y - by + 1) * scale);

        // scaling factor for part filters
        scale = SIDE_LENGTH * powf(step, (float)(levels[i] - lambda - initialImageLevel));
        for (j = 0; j < n; j++)
        {
            partsDisplacement[i][j].x = (int)((partsDisplacement[i][j].x -
                                               2 * bx + 1) * scale);
            partsDisplacement[i][j].y = (int)((partsDisplacement[i][j].y -
                                               2 * by + 1) * scale);
        }
    }
    return LATENT_SVM_OK;
}

/*
// Elimination boxes that are outside the image boudaries
//
// API
// int clippingBoxes(int width, int height,
                     CvPoint *points, int kPoints);
// INPUT
// width             - image wediht
// height            - image heigth
// points            - a set of points (coordinates of top left or
                       bottom right corners)
// kPoints           - points number
// OUTPUT
// points            - updated points (if coordinates less than zero then
                       set zero coordinate, if coordinates more than image
                       size then set coordinates equal image size)
// RESULT
// Error status
*/
int clippingBoxes(int width, int height,
                  CvPoint *points, int kPoints)
{
    int i;
    for (i = 0; i < kPoints; i++)
    {
        if (points[i].x > width - 1)
        {
            points[i].x = width - 1;
        }
        if (points[i].x < 0)
        {
            points[i].x = 0;
        }
        if (points[i].y > height - 1)
        {
            points[i].y = height - 1;
        }
        if (points[i].y < 0)
        {
            points[i].y = 0;
        }
    }
    return LATENT_SVM_OK;
}

/*
// Computation of the root filter displacement and values of score function
//
// API
// int searchObject(const featurePyramid *H, const filterObject **all_F, int n,
                    float b,
                    int maxXBorder,
                     int maxYBorder,
                     CvPoint **points, int **levels, int *kPoints, float *score,
                     CvPoint ***partsDisplacement);
// INPUT
// image             - initial image for searhing object
// all_F             - the set of filters (the first element is root filter,
                       other elements - part filters)
// n                 - the number of part filters
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// points            - positions (x, y) of the upper-left corner
                       of root filter frame
// levels            - levels that correspond to each position
// kPoints           - number of positions
// score             - value of the score function
// partsDisplacement - part filters displacement for each position
                       of the root filter
// RESULT
// Error status
*/
int searchObject(const CvLSVMFeaturePyramid *H, const CvLSVMFilterObject **all_F,
                 int n, float b,
                 int maxXBorder,
                 int maxYBorder,
                 CvPoint **points, int **levels, int *kPoints, float *score,
                 CvPoint ***partsDisplacement)
{
    int opResult;

    // Matching
    opResult = maxFunctionalScore(all_F, n, H, b, maxXBorder, maxYBorder,
                                  score, points, levels,
                                  kPoints, partsDisplacement);
    if (opResult != LATENT_SVM_OK)
    {
        return LATENT_SVM_SEARCH_OBJECT_FAILED;
    }

    // Transformation filter displacement from the block space
    // to the space of pixels at the initial image
    // that settles at the level number LAMBDA
    convertPoints( H->numLevels, LAMBDA, LAMBDA, (*points),
                   (*levels), (*partsDisplacement), (*kPoints), n,
                   maxXBorder, maxYBorder );

    return LATENT_SVM_OK;
}

/*
// Computation right bottom corners coordinates of bounding boxes
//
// API
// int estimateBoxes(CvPoint *points, int *levels, int kPoints,
                     int sizeX, int sizeY, CvPoint **oppositePoints);
// INPUT
// points            - left top corners coordinates of bounding boxes
// levels            - levels of feature pyramid where points were found
// (sizeX, sizeY)    - size of root filter
// OUTPUT
// oppositePoins     - right bottom corners coordinates of bounding boxes
// RESULT
// Error status
*/
static int estimateBoxes(CvPoint *points, int *levels, int kPoints,
                  int sizeX, int sizeY, CvPoint **oppositePoints)
{
    int i;
    float step;

    step = powf( 2.0f, 1.0f / ((float)(LAMBDA)));

    *oppositePoints = (CvPoint *)malloc(sizeof(CvPoint) * kPoints);
    for (i = 0; i < kPoints; i++)
    {
        getOppositePoint(points[i], sizeX, sizeY, step, levels[i] - LAMBDA, &((*oppositePoints)[i]));
    }
    return LATENT_SVM_OK;
}

/*
// Computation maximum filter size for each dimension
//
// API
// int getMaxFilterDims(const filterObject **filters, int kComponents,
                        const int *kPartFilters,
                        unsigned int *maxXBorder, unsigned int *maxYBorder);
// INPUT
// filters           - a set of filters (at first root filter, then part filters
                       and etc. for all components)
// kComponents       - number of components
// kPartFilters      - number of part filters for each component
// OUTPUT
// maxXBorder        - maximum of filter size at the horizontal dimension
// maxYBorder        - maximum of filter size at the vertical dimension
// RESULT
// Error status
*/
int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
                     const int *kPartFilters,
                     unsigned int *maxXBorder, unsigned int *maxYBorder)
{
    int i, componentIndex;
    *maxXBorder = filters[0]->sizeX;
    *maxYBorder = filters[0]->sizeY;
    componentIndex = kPartFilters[0] + 1;
    for (i = 1; i < kComponents; i++)
    {
        if ((unsigned)filters[componentIndex]->sizeX > *maxXBorder)
        {
            *maxXBorder = filters[componentIndex]->sizeX;
        }
        if ((unsigned)filters[componentIndex]->sizeY > *maxYBorder)
        {
            *maxYBorder = filters[componentIndex]->sizeY;
        }
        componentIndex += (kPartFilters[i] + 1);
    }
    return LATENT_SVM_OK;
}

/*
// Perform non-maximum suppression algorithm (described in original paper)
// to remove "similar" bounding boxes
//
// API
// int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                             const CvPoint *oppositePoints, const float *score,
                             float overlapThreshold,
                             int *numBoxesOut, CvPoint **pointsOut,
                             CvPoint **oppositePointsOut, float **scoreOut);
// INPUT
// numBoxes          - number of bounding boxes
// points            - array of left top corner coordinates
// oppositePoints    - array of right bottom corner coordinates
// score             - array of detection scores
// overlapThreshold  - threshold: bounding box is removed if overlap part
                       is greater than passed value
// OUTPUT
// numBoxesOut       - the number of bounding boxes algorithm returns
// pointsOut         - array of left top corner coordinates
// oppositePointsOut - array of right bottom corner coordinates
// scoreOut          - array of detection scores
// RESULT
// Error status
*/
int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                          const CvPoint *oppositePoints, const float *score,
                          float overlapThreshold,
                          int *numBoxesOut, CvPoint **pointsOut,
                          CvPoint **oppositePointsOut, float **scoreOut)
{
    int i, j, index;
    float* box_area = (float*)malloc(numBoxes * sizeof(float));
    int* indices = (int*)malloc(numBoxes * sizeof(int));
    int* is_suppressed = (int*)malloc(numBoxes * sizeof(int));

    for (i = 0; i < numBoxes; i++)
    {
        indices[i] = i;
        is_suppressed[i] = 0;
        box_area[i] = (float)( (oppositePoints[i].x - points[i].x + 1) *
                                (oppositePoints[i].y - points[i].y + 1));
    }

    sort(numBoxes, score, indices);
    for (i = 0; i < numBoxes; i++)
    {
        if (!is_suppressed[indices[i]])
        {
            for (j = i + 1; j < numBoxes; j++)
            {
                if (!is_suppressed[indices[j]])
                {
                    int x1max = max(points[indices[i]].x, points[indices[j]].x);
                    int x2min = min(oppositePoints[indices[i]].x, oppositePoints[indices[j]].x);
                    int y1max = max(points[indices[i]].y, points[indices[j]].y);
                    int y2min = min(oppositePoints[indices[i]].y, oppositePoints[indices[j]].y);
                    int overlapWidth = x2min - x1max + 1;
                    int overlapHeight = y2min - y1max + 1;
                    if (overlapWidth > 0 && overlapHeight > 0)
                    {
                        float overlapPart = (overlapWidth * overlapHeight) / box_area[indices[j]];
                        if (overlapPart > overlapThreshold)
                        {
                            is_suppressed[indices[j]] = 1;
                        }
                    }
                }
            }
        }
    }

    *numBoxesOut = 0;
    for (i = 0; i < numBoxes; i++)
    {
        if (!is_suppressed[i]) (*numBoxesOut)++;
    }

    *pointsOut = (CvPoint *)malloc((*numBoxesOut) * sizeof(CvPoint));
    *oppositePointsOut = (CvPoint *)malloc((*numBoxesOut) * sizeof(CvPoint));
    *scoreOut = (float *)malloc((*numBoxesOut) * sizeof(float));
    index = 0;
    for (i = 0; i < numBoxes; i++)
    {
        if (!is_suppressed[indices[i]])
        {
            (*pointsOut)[index].x = points[indices[i]].x;
            (*pointsOut)[index].y = points[indices[i]].y;
            (*oppositePointsOut)[index].x = oppositePoints[indices[i]].x;
            (*oppositePointsOut)[index].y = oppositePoints[indices[i]].y;
            (*scoreOut)[index] = score[indices[i]];
            index++;
        }

    }

    free(indices);
    free(box_area);
    free(is_suppressed);

    return LATENT_SVM_OK;
}

/*
// Computation root filters displacement and values of score function
//
// API
// int searchObjectThresholdSomeComponents(const featurePyramid *H,
                                           const filterObject **filters,
                                           int kComponents, const int *kPartFilters,
                                           const float *b, float scoreThreshold,
                                           CvPoint **points, CvPoint **oppPoints,
                                           float **score, int *kPoints);
// INPUT
// H                 - feature pyramid
// filters           - filters (root filter then it's part filters, etc.)
// kComponents       - root filters number
// kPartFilters      - array of part filters number for each component
// b                 - array of linear terms
// scoreThreshold    - score threshold
// OUTPUT
// points            - root filters displacement (top left corners)
// oppPoints         - root filters displacement (bottom right corners)
// score             - array of score values
// kPoints           - number of boxes
// RESULT
// Error status
*/
int searchObjectThresholdSomeComponents(const CvLSVMFeaturePyramid *H,
                                        const CvLSVMFilterObject **filters,
                                        int kComponents, const int *kPartFilters,
                                        const float *b, float scoreThreshold,
                                        CvPoint **points, CvPoint **oppPoints,
                                        float **score, int *kPoints,
                                        int numThreads)
{
    //int error = 0;
    int i, j, s, f, componentIndex;
    unsigned int maxXBorder, maxYBorder;
    CvPoint **pointsArr, **oppPointsArr, ***partsDisplacementArr;
    float **scoreArr;
    int *kPointsArr, **levelsArr;

    // Allocation memory
    pointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
    oppPointsArr = (CvPoint **)malloc(sizeof(CvPoint *) * kComponents);
    scoreArr = (float **)malloc(sizeof(float *) * kComponents);
    kPointsArr = (int *)malloc(sizeof(int) * kComponents);
    levelsArr = (int **)malloc(sizeof(int *) * kComponents);
    partsDisplacementArr = (CvPoint ***)malloc(sizeof(CvPoint **) * kComponents);

    // Getting maximum filter dimensions
    /*error = */getMaxFilterDims(filters, kComponents, kPartFilters, &maxXBorder, &maxYBorder);
    componentIndex = 0;
    *kPoints = 0;
    // For each component perform searching
    for (i = 0; i < kComponents; i++)
    {
        int error = searchObjectThreshold(H, &(filters[componentIndex]), kPartFilters[i],
            b[i], maxXBorder, maxYBorder, scoreThreshold,
            &(pointsArr[i]), &(levelsArr[i]), &(kPointsArr[i]),
            &(scoreArr[i]), &(partsDisplacementArr[i]), numThreads);
        if (error != LATENT_SVM_OK)
        {
            // Release allocated memory
            free(pointsArr);
            free(oppPointsArr);
            free(scoreArr);
            free(kPointsArr);
            free(levelsArr);
            free(partsDisplacementArr);
            return LATENT_SVM_SEARCH_OBJECT_FAILED;
        }
        estimateBoxes(pointsArr[i], levelsArr[i], kPointsArr[i],
            filters[componentIndex]->sizeX, filters[componentIndex]->sizeY, &(oppPointsArr[i]));
        componentIndex += (kPartFilters[i] + 1);
        *kPoints += kPointsArr[i];
    }

    *points = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    *oppPoints = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    *score = (float *)malloc(sizeof(float) * (*kPoints));
    s = 0;
    for (i = 0; i < kComponents; i++)
    {
        f = s + kPointsArr[i];
        for (j = s; j < f; j++)
        {
            (*points)[j].x = pointsArr[i][j - s].x;
            (*points)[j].y = pointsArr[i][j - s].y;
            (*oppPoints)[j].x = oppPointsArr[i][j - s].x;
            (*oppPoints)[j].y = oppPointsArr[i][j - s].y;
            (*score)[j] = scoreArr[i][j - s];
        }
        s = f;
    }

    // Release allocated memory
    for (i = 0; i < kComponents; i++)
    {
        free(pointsArr[i]);
        free(oppPointsArr[i]);
        free(scoreArr[i]);
        free(levelsArr[i]);
        for (j = 0; j < kPointsArr[i]; j++)
        {
            free(partsDisplacementArr[i][j]);
        }
        free(partsDisplacementArr[i]);
    }
    free(pointsArr);
    free(oppPointsArr);
    free(scoreArr);
    free(kPointsArr);
    free(levelsArr);
    free(partsDisplacementArr);
    return LATENT_SVM_OK;
}

/*
// Computation the maximum of the score function
//
// API
// int maxFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                          const featurePyramid *H, float b,
                          int maxXBorder, int maxYBorder,
                          float *score,
                          CvPoint **points, int **levels, int *kPoints,
                          CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// score             - the maximum of the score function
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int maxFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                       const CvLSVMFeaturePyramid *H, float b,
                       int maxXBorder, int maxYBorder,
                       float *score,
                       CvPoint **points, int **levels, int *kPoints,
                       CvPoint ***partsDisplacement)
{
    int l, i, j, k, s, f, level, numLevels;
    float *tmpScore;
    CvPoint ***tmpPoints;
    CvPoint ****tmpPartsDisplacement;
    int *tmpKPoints;
    float maxScore;
    int res;

    /* DEBUG
    FILE *file;
    //*/

    // Computation the number of levels for seaching object,
    // first lambda-levels are used for computation values
    // of score function for each position of root filter
    numLevels = H->numLevels - LAMBDA;

    // Allocation memory for maximum value of score function for each level
    tmpScore = (float *)malloc(sizeof(float) * numLevels);
    // Allocation memory for the set of points that corresponds
    // to the maximum of score function
    tmpPoints = (CvPoint ***)malloc(sizeof(CvPoint **) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPoints[i] = (CvPoint **)malloc(sizeof(CvPoint *));
    }
    // Allocation memory for memory for saving parts displacement on each level
    tmpPartsDisplacement = (CvPoint ****)malloc(sizeof(CvPoint ***) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPartsDisplacement[i] = (CvPoint ***)malloc(sizeof(CvPoint **));
    }
    // Number of points that corresponds to the maximum
    // of score function on each level
    tmpKPoints = (int *)malloc(sizeof(int) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpKPoints[i] = 0;
    }

    // Set current value of the maximum of score function
    res = maxFunctionalScoreFixedLevel(all_F, n, H, LAMBDA, b,
            maxXBorder, maxYBorder,
            &(tmpScore[0]),
            tmpPoints[0],
            &(tmpKPoints[0]),
            tmpPartsDisplacement[0]);
    maxScore = tmpScore[0];
    (*kPoints) = tmpKPoints[0];

    // Computation maxima of score function on each level
    // and getting the maximum on all levels
    /* DEBUG: maxScore
    file = fopen("maxScore.csv", "w+");
    fprintf(file, "%i;%lf;\n", H->lambda, tmpScore[0]);
    //*/
    for (l = LAMBDA + 1; l < H->numLevels; l++)
    {
        k = l - LAMBDA;
        res = maxFunctionalScoreFixedLevel(all_F, n, H, l, b,
                                           maxXBorder, maxYBorder,
                                           &(tmpScore[k]),
                                           tmpPoints[k],
                                           &(tmpKPoints[k]),
                                           tmpPartsDisplacement[k]);
        //fprintf(file, "%i;%lf;\n", l, tmpScore[k]);
        if (res != LATENT_SVM_OK)
        {
            continue;
        }
        if (maxScore < tmpScore[k])
        {
            maxScore = tmpScore[k];
            (*kPoints) = tmpKPoints[k];
        }
        else if ((maxScore - tmpScore[k]) * (maxScore - tmpScore[k]) <= EPS)
        {
            (*kPoints) += tmpKPoints[k];
        } /* if (maxScore < tmpScore[k]) else if (...)*/
    }
    //fclose(file);

    // Allocation memory for levels
    (*levels) = (int *)malloc(sizeof(int) * (*kPoints));
    // Allocation memory for the set of points
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    // Allocation memory for parts displacement
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));

    // Filling the set of points, levels and parts displacement
    s = 0;
    f = 0;
    for (i = 0; i < numLevels; i++)
    {
        if ((tmpScore[i] - maxScore) * (tmpScore[i] - maxScore) <= EPS)
        {
            // Computation the number of level
            level = i + LAMBDA;

            // Addition a set of points
            f += tmpKPoints[i];
            for (j = s; j < f; j++)
            {
                (*levels)[j] = level;
                (*points)[j] = (*tmpPoints[i])[j - s];
                (*partsDisplacement)[j] = (*(tmpPartsDisplacement[i]))[j - s];
            }
            s = f;
        } /* if ((tmpScore[i] - maxScore) * (tmpScore[i] - maxScore) <= EPS) */
    }
    (*score) = maxScore;

    // Release allocated memory
    for (i = 0; i < numLevels; i++)
    {
        free(tmpPoints[i]);
        free(tmpPartsDisplacement[i]);
    }
    free(tmpPoints);
    free(tmpPartsDisplacement);
    free(tmpScore);
    free(tmpKPoints);

    return LATENT_SVM_OK;
}

/*
// Compute opposite point for filter box
//
// API
// int getOppositePoint(CvPoint point,
                        int sizeX, int sizeY,
                        float step, int degree,
                        CvPoint *oppositePoint);

// INPUT
// point             - coordinates of filter top left corner
                       (in the space of pixels)
// (sizeX, sizeY)    - filter dimension in the block space
// step              - scaling factor
// degree            - degree of the scaling factor
// OUTPUT
// oppositePoint     - coordinates of filter bottom corner
                       (in the space of pixels)
// RESULT
// Error status
*/
int getOppositePoint(CvPoint point,
                     int sizeX, int sizeY,
                     float step, int degree,
                     CvPoint *oppositePoint)
{
    float scale;
    scale = SIDE_LENGTH * powf(step, (float)degree);
    oppositePoint->x = (int)(point.x + sizeX * scale);
    oppositePoint->y = (int)(point.y + sizeY * scale);
    return LATENT_SVM_OK;
}

/*
// Computation of the root filter displacement and values of score function
//
// API
// int searchObjectThreshold(const featurePyramid *H,
                             const filterObject **all_F, int n,
                             float b,
                             int maxXBorder, int maxYBorder,
                             float scoreThreshold,
                             CvPoint **points, int **levels, int *kPoints,
                             float **score, CvPoint ***partsDisplacement);
// INPUT
// H                 - feature pyramid
// all_F             - the set of filters (the first element is root filter,
                       other elements - part filters)
// n                 - the number of part filters
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// points            - positions (x, y) of the upper-left corner
                       of root filter frame
// levels            - levels that correspond to each position
// kPoints           - number of positions
// score             - values of the score function
// partsDisplacement - part filters displacement for each position
                       of the root filter
// RESULT
// Error status
*/
int searchObjectThreshold(const CvLSVMFeaturePyramid *H,
                          const CvLSVMFilterObject **all_F, int n,
                          float b,
                          int maxXBorder, int maxYBorder,
                          float scoreThreshold,
                          CvPoint **points, int **levels, int *kPoints,
                          float **score, CvPoint ***partsDisplacement,
                          int numThreads)
{
    int opResult;


    opResult = thresholdFunctionalScore(all_F, n, H, b,
                                        maxXBorder, maxYBorder,
                                        scoreThreshold,
                                        score, points, levels,
                                        kPoints, partsDisplacement);

  (void)numThreads;

    if (opResult != LATENT_SVM_OK)
    {
        return LATENT_SVM_SEARCH_OBJECT_FAILED;
    }

    // Transformation filter displacement from the block space
    // to the space of pixels at the initial image
    // that settles at the level number LAMBDA
    convertPoints(H->numLevels, LAMBDA, LAMBDA, (*points),
                  (*levels), (*partsDisplacement), (*kPoints), n,
                  maxXBorder, maxYBorder);

    return LATENT_SVM_OK;
}

/*
// Computation the maximum of the score function at the level
//
// API
// int maxFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                    const featurePyramid *H,
                                    int level, float b,
                                    int maxXBorder, int maxYBorder,
                                    float *score, CvPoint **points, int *kPoints,
                                    CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// level             - feature pyramid level for computation maximum score
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// score             - the maximum of the score function at the level
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int maxFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                 const CvLSVMFeaturePyramid *H,
                                 int level, float b,
                                 int maxXBorder, int maxYBorder,
                                 float *score, CvPoint **points,
                                 int *kPoints, CvPoint ***partsDisplacement)
{
    int i, j, k, dimX, dimY, nF0, mF0/*, p*/;
    int diff1, diff2, index, last, partsLevel;
    CvLSVMFilterDisposition **disposition;
    float *f;
    float *scores;
    float sumScorePartDisposition, maxScore;
    int res;
    CvLSVMFeatureMap *map;

    /*
    // DEBUG variables
    FILE *file;
    char *tmp;
    char buf[40] = "..\\Data\\score\\score", buf1[10] = ".csv";
    tmp = (char *)malloc(sizeof(char) * 80);
    itoa(level, tmp, 10);
    strcat(tmp, buf1);
    //*/

    // Feature map matrix dimension on the level
    dimX = H->pyramid[level]->sizeX;
    dimY = H->pyramid[level]->sizeY;

    // Number of features
    //p = H->pyramid[level]->numFeatures;

    // Getting dimension of root filter
    nF0 = all_F[0]->sizeY;
    mF0 = all_F[0]->sizeX;
    // Processing the situation when root filter goes
    // beyond the boundaries of the block set
    if (nF0 > dimY || mF0 > dimX)
    {
        return LATENT_SVM_FAILED_SUPERPOSITION;
    }

    diff1 = dimY - nF0 + 1;
    diff2 = dimX - mF0 + 1;

    // Allocation memory for saving values of function D
    // on the level for each part filter
    disposition = (CvLSVMFilterDisposition **)malloc(sizeof(CvLSVMFilterDisposition *) * n);
    for (i = 0; i < n; i++)
    {
        disposition[i] = (CvLSVMFilterDisposition *)malloc(sizeof(CvLSVMFilterDisposition));
    }

    // Allocation memory for values of score function for each block on the level
    scores = (float *)malloc(sizeof(float) * (diff1 * diff2));

    // A dot product vectors of feature map and weights of root filter

    // Allocation memory for saving a dot product vectors of feature map and
    // weights of root filter
    f = (float *)malloc(sizeof(float) * (diff1 * diff2));
    // A dot product vectors of feature map and weights of root filter
    res = convolution(all_F[0], H->pyramid[level], f);
    
    if (res != LATENT_SVM_OK)
    {
        free(f);
        free(scores);
        for (i = 0; i < n; i++)
        {
            free(disposition[i]);
        }
        free(disposition);
        return res;
    }

    // Computation values of function D for each part filter
    // on the level (level - LAMBDA)
    partsLevel = level - LAMBDA;
    // For feature map at the level 'partsLevel' add nullable border
    map = featureMapBorderPartFilter(H->pyramid[partsLevel],
                                     maxXBorder, maxYBorder);

    // Computation the maximum of score function
    sumScorePartDisposition = 0.0;
    for (k = 1; k <= n; k++)
    {
        filterDispositionLevel(all_F[k], map,
                               &(disposition[k - 1]->score),
                               &(disposition[k - 1]->x),
                               &(disposition[k - 1]->y));
    }

    scores[0] = f[0] - sumScorePartDisposition + b;
    maxScore = scores[0];
    (*kPoints) = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            sumScorePartDisposition = 0.0;
            for (k = 1; k <= n; k++)
            {
                // This condition takes on a value true
                // when filter goes beyond the boundaries of block set
                if ((2 * i + all_F[k]->V.y <
                            map->sizeY - all_F[k]->sizeY + 1) &&
                    (2 * j + all_F[k]->V.x <
                            map->sizeX - all_F[k]->sizeX + 1))
                {
                    index = (2 * i + all_F[k]->V.y) *
                                (map->sizeX - all_F[k]->sizeX + 1) +
                            (2 * j + all_F[k]->V.x);
                    sumScorePartDisposition += disposition[k - 1]->score[index];
                }
            }
            scores[i * diff2 + j] = f[i * diff2 + j] - sumScorePartDisposition + b;
            if (maxScore < scores[i * diff2 + j])
            {
                maxScore = scores[i * diff2 + j];
                (*kPoints) = 1;
            }
            else if ((scores[i * diff2 + j] - maxScore) *
                     (scores[i * diff2 + j] - maxScore) <= EPS)
            {
                (*kPoints)++;
            } /* if (maxScore < scores[i * diff2 + j]) */
        }
    }

    // Allocation memory for saving positions of root filter and part filters
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
    for (i = 0; i < (*kPoints); i++)
    {
        (*partsDisplacement)[i] = (CvPoint *)malloc(sizeof(CvPoint) * n);
    }

    /*// DEBUG
    strcat(buf, tmp);
    file = fopen(buf, "w+");
    //*/
    // Construction of the set of positions for root filter
    // that correspond the maximum of score function on the level
    (*score) = maxScore;
    last = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            if ((scores[i * diff2 + j] - maxScore) *
                (scores[i * diff2 + j] - maxScore) <= EPS)
            {
                (*points)[last].y = i;
                (*points)[last].x = j;
                for (k = 1; k <= n; k++)
                {
                    if ((2 * i + all_F[k]->V.y <
                            map->sizeY - all_F[k]->sizeY + 1) &&
                        (2 * j + all_F[k]->V.x <
                            map->sizeX - all_F[k]->sizeX + 1))
                    {
                        index = (2 * i + all_F[k]->V.y) *
                                   (map->sizeX - all_F[k]->sizeX + 1) +
                                (2 * j + all_F[k]->V.x);
                        (*partsDisplacement)[last][k - 1].x =
                                              disposition[k - 1]->x[index];
                        (*partsDisplacement)[last][k - 1].y =
                                              disposition[k - 1]->y[index];
                    }
                }
                last++;
            } /* if ((scores[i * diff2 + j] - maxScore) *
                     (scores[i * diff2 + j] - maxScore) <= EPS) */
            //fprintf(file, "%lf;", scores[i * diff2 + j]);
        }
        //fprintf(file, "\n");
    }
    //fclose(file);
    //free(tmp);

    // Release allocated memory
    for (i = 0; i < n ; i++)
    {
        free(disposition[i]->score);
        free(disposition[i]->x);
        free(disposition[i]->y);
        free(disposition[i]);
    }
    free(disposition);
    free(f);
    free(scores);
    freeFeatureMapObject(&map);
    return LATENT_SVM_OK;
}

/*
// Function for convolution computation
//
// INPUT
// Fi                - filter object
// map               - feature map
// OUTPUT
// f                 - the convolution
// RESULT
// Error status
*/
int convolution(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *map, float *f)
{
    int n1, m1, n2, m2, p, /*size,*/ diff1, diff2;
    int i1, i2, j1, j2, k;
    float tmp_f1, tmp_f2, tmp_f3, tmp_f4;
    float *pMap = NULL;
    float *pH = NULL;

    n1 = map->sizeY;
    m1 = map->sizeX;
    n2 = Fi->sizeY;
    m2 = Fi->sizeX;
    p = map->numFeatures;

    diff1 = n1 - n2 + 1;
    diff2 = m1 - m2 + 1;
    //size = diff1 * diff2;
    for (j1 = diff2 - 1; j1 >= 0; j1--)
    {

        for (i1 = diff1 - 1; i1 >= 0; i1--)
        {
            tmp_f1 = 0.0f;
            tmp_f2 = 0.0f;
            tmp_f3 = 0.0f;
            tmp_f4 = 0.0f;
            for (i2 = 0; i2 < n2; i2++)
            {
                for (j2 = 0; j2 < m2; j2++)
                {
                    pMap = map->map + (i1 + i2) * m1 * p + (j1 + j2) * p;//sm2
                    pH = Fi->H + (i2 * m2 + j2) * p;//sm2
                    for (k = 0; k < p/4; k++)
                    {

                        tmp_f1 += pMap[4*k]*pH[4*k];//sm2
                        tmp_f2 += pMap[4*k+1]*pH[4*k+1];
                        tmp_f3 += pMap[4*k+2]*pH[4*k+2];
                        tmp_f4 += pMap[4*k+3]*pH[4*k+3];
                    }

                    if (p%4==1)
                    {
                        tmp_f1 += pH[p-1]*pMap[p-1];
                    }
                    else
                    {
                        if (p%4==2)
                        {
                            tmp_f1 += pH[p-2]*pMap[p-2] + pH[p-1]*pMap[p-1];
                        }
                        else
                        {
                            if (p%4==3)
                            {
                                tmp_f1 += pH[p-3]*pMap[p-3] + pH[p-2]*pMap[p-2] + pH[p-1]*pMap[p-1];
                            }
                        }
                    }

                }
            }
            f[i1 * diff2 + j1] = tmp_f1 + tmp_f2 + tmp_f3 + tmp_f4;//sm1
        }
    }
    return LATENT_SVM_OK;
}

/*
// Computation score function that exceed threshold
//
// API
// int thresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                                const featurePyramid *H,
                                float b,
                                int maxXBorder, int maxYBorder,
                                float scoreThreshold,
                                float **score,
                                CvPoint **points, int **levels, int *kPoints,
                                CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// score             - score function values that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int thresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                             const CvLSVMFeaturePyramid *H,
                             float b,
                             int maxXBorder, int maxYBorder,
                             float scoreThreshold,
                             float **score,
                             CvPoint **points, int **levels, int *kPoints,
                             CvPoint ***partsDisplacement)
{
    int l, i, j, k, s, f, level, numLevels;
    float **tmpScore;
    CvPoint ***tmpPoints;
    CvPoint ****tmpPartsDisplacement;
    int *tmpKPoints;
    int res;

    /* DEBUG
    FILE *file;
    //*/

    // Computation the number of levels for seaching object,
    // first lambda-levels are used for computation values
    // of score function for each position of root filter
    numLevels = H->numLevels - LAMBDA;

    // Allocation memory for values of score function for each level
    // that exceed threshold
    tmpScore = (float **)malloc(sizeof(float*) * numLevels);
    // Allocation memory for the set of points that corresponds
    // to the maximum of score function
    tmpPoints = (CvPoint ***)malloc(sizeof(CvPoint **) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPoints[i] = (CvPoint **)malloc(sizeof(CvPoint *));
    }
    // Allocation memory for memory for saving parts displacement on each level
    tmpPartsDisplacement = (CvPoint ****)malloc(sizeof(CvPoint ***) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpPartsDisplacement[i] = (CvPoint ***)malloc(sizeof(CvPoint **));
    }
    // Number of points that corresponds to the maximum
    // of score function on each level
    tmpKPoints = (int *)malloc(sizeof(int) * numLevels);
    for (i = 0; i < numLevels; i++)
    {
        tmpKPoints[i] = 0;
    }

    // Computation maxima of score function on each level
    // and getting the maximum on all levels
    /* DEBUG: maxScore
    file = fopen("maxScore.csv", "w+");
    fprintf(file, "%i;%lf;\n", H->lambda, tmpScore[0]);
    //*/
    (*kPoints) = 0;
    for (l = LAMBDA; l < H->numLevels; l++)
    {
        k = l - LAMBDA;
        //printf("Score at the level %i\n", l);
        res = thresholdFunctionalScoreFixedLevel(all_F, n, H, l, b,
            maxXBorder, maxYBorder, scoreThreshold,
            &(tmpScore[k]),
            tmpPoints[k],
            &(tmpKPoints[k]),
            tmpPartsDisplacement[k]);
        //fprintf(file, "%i;%lf;\n", l, tmpScore[k]);
        if (res != LATENT_SVM_OK)
        {
            continue;
        }
        (*kPoints) += tmpKPoints[k];
    }

    //fclose(file);

    // Allocation memory for levels
    (*levels) = (int *)malloc(sizeof(int) * (*kPoints));
    // Allocation memory for the set of points
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    // Allocation memory for parts displacement
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
    // Allocation memory for score function values
    (*score) = (float *)malloc(sizeof(float) * (*kPoints));

    // Filling the set of points, levels and parts displacement
    s = 0;
    f = 0;
    for (i = 0; i < numLevels; i++)
    {
        // Computation the number of level
        level = i + LAMBDA;

        // Addition a set of points
        f += tmpKPoints[i];
        for (j = s; j < f; j++)
        {
            (*levels)[j] = level;
            (*points)[j] = (*tmpPoints[i])[j - s];
            (*score)[j] = tmpScore[i][j - s];
            (*partsDisplacement)[j] = (*(tmpPartsDisplacement[i]))[j - s];
        }
        s = f;
    }

    // Release allocated memory
    for (i = 0; i < numLevels; i++)
    {
        free(tmpPoints[i]);
        free(tmpPartsDisplacement[i]);
    }
    free(tmpPoints);
    free(tmpScore);
    free(tmpKPoints);
    free(tmpPartsDisplacement);

    return LATENT_SVM_OK;
}

/*
// Computation score function at the level that exceed threshold
//
// API
// int thresholdFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                          const featurePyramid *H,
                                          int level, float b,
                                          int maxXBorder, int maxYBorder,
                                          float scoreThreshold,
                                          float **score, CvPoint **points, int *kPoints,
                                          CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// level             - feature pyramid level for computation maximum score
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// score             - score function at the level that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int thresholdFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                       const CvLSVMFeaturePyramid *H,
                                       int level, float b,
                                       int maxXBorder, int maxYBorder,
                                       float scoreThreshold,
                                       float **score, CvPoint **points, int *kPoints,
                                       CvPoint ***partsDisplacement)
{
    int i, j, k, dimX, dimY, nF0, mF0/*, p*/;
    int diff1, diff2, index, last, partsLevel;
    CvLSVMFilterDisposition **disposition;
    float *f;
    float *scores;
    float sumScorePartDisposition;
    int res;
    CvLSVMFeatureMap *map;

    /*
    // DEBUG variables
    FILE *file;
    char *tmp;
    char buf[40] = "..\\Data\\score\\score", buf1[10] = ".csv";
    tmp = (char *)malloc(sizeof(char) * 80);
    itoa(level, tmp, 10);
    strcat(tmp, buf1);
    //*/

    // Feature map matrix dimension on the level
    dimX = H->pyramid[level]->sizeX;
    dimY = H->pyramid[level]->sizeY;

    // Number of features
    //p = H->pyramid[level]->numFeatures;

    // Getting dimension of root filter
    nF0 = all_F[0]->sizeY;
    mF0 = all_F[0]->sizeX;
    // Processing the situation when root filter goes
    // beyond the boundaries of the block set
    if (nF0 > dimY || mF0 > dimX)
    {
        return LATENT_SVM_FAILED_SUPERPOSITION;
    }

    diff1 = dimY - nF0 + 1;
    diff2 = dimX - mF0 + 1;

    // Allocation memory for saving values of function D
    // on the level for each part filter
    disposition = (CvLSVMFilterDisposition **)malloc(sizeof(CvLSVMFilterDisposition *) * n);
    for (i = 0; i < n; i++)
    {
        disposition[i] = (CvLSVMFilterDisposition *)malloc(sizeof(CvLSVMFilterDisposition));
    }

    // Allocation memory for values of score function for each block on the level
    scores = (float *)malloc(sizeof(float) * (diff1 * diff2));
    // A dot product vectors of feature map and weights of root filter

    // Allocation memory for saving a dot product vectors of feature map and
    // weights of root filter

    f = (float *)malloc(sizeof(float) * (diff1 * diff2));
    res = convolution( all_F[0], H->pyramid[level], f );

    if (res != LATENT_SVM_OK)
    {
        free(f);
        free(scores);
        for (i = 0; i < n; i++)
        {
            free(disposition[i]);
        }
        free(disposition);
        return res;
    }

    // Computation values of function D for each part filter
    // on the level (level - LAMBDA)
    partsLevel = level - LAMBDA;
    // For feature map at the level 'partsLevel' add nullable border
    map = featureMapBorderPartFilter(H->pyramid[partsLevel],
                                     maxXBorder, maxYBorder);

    // Computation the maximum of score function
    sumScorePartDisposition = 0.0;

    for (k = 1; k <= n; k++)
    {
        filterDispositionLevel(all_F[k], map,
                               &(disposition[k - 1]->score),
                               &(disposition[k - 1]->x),
                               &(disposition[k - 1]->y));
    }

    (*kPoints) = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            sumScorePartDisposition = 0.0;
            for (k = 1; k <= n; k++)
            {
                // This condition takes on a value true
                // when filter goes beyond the boundaries of block set
                if ((2 * i + all_F[k]->V.y <
                            map->sizeY - all_F[k]->sizeY + 1) &&
                    (2 * j + all_F[k]->V.x <
                            map->sizeX - all_F[k]->sizeX + 1))
                {
                    index = (2 * i + all_F[k]->V.y) *
                                (map->sizeX - all_F[k]->sizeX + 1) +
                            (2 * j + all_F[k]->V.x);
                    sumScorePartDisposition += disposition[k - 1]->score[index];
                }
            }
            scores[i * diff2 + j] = f[i * diff2 + j] - sumScorePartDisposition + b;
            if (scores[i * diff2 + j] > scoreThreshold)
            {
                (*kPoints)++;
            }
        }
    }

    // Allocation memory for saving positions of root filter and part filters
    (*points) = (CvPoint *)malloc(sizeof(CvPoint) * (*kPoints));
    (*partsDisplacement) = (CvPoint **)malloc(sizeof(CvPoint *) * (*kPoints));
    for (i = 0; i < (*kPoints); i++)
    {
        (*partsDisplacement)[i] = (CvPoint *)malloc(sizeof(CvPoint) * n);
    }

    /*// DEBUG
    strcat(buf, tmp);
    file = fopen(buf, "w+");
    //*/
    // Construction of the set of positions for root filter
    // that correspond score function on the level that exceed threshold
    (*score) = (float *)malloc(sizeof(float) * (*kPoints));
    last = 0;
    for (i = 0; i < diff1; i++)
    {
        for (j = 0; j < diff2; j++)
        {
            if (scores[i * diff2 + j] > scoreThreshold)
            {
                (*score)[last] = scores[i * diff2 + j];
                (*points)[last].y = i;
                (*points)[last].x = j;
                for (k = 1; k <= n; k++)
                {
                    if ((2 * i + all_F[k]->V.y <
                            map->sizeY - all_F[k]->sizeY + 1) &&
                        (2 * j + all_F[k]->V.x <
                            map->sizeX - all_F[k]->sizeX + 1))
                    {
                        index = (2 * i + all_F[k]->V.y) *
                                   (map->sizeX - all_F[k]->sizeX + 1) +
                                (2 * j + all_F[k]->V.x);
                        (*partsDisplacement)[last][k - 1].x =
                                              disposition[k - 1]->x[index];
                        (*partsDisplacement)[last][k - 1].y =
                                              disposition[k - 1]->y[index];
                    }
                }
                last++;
            }
            //fprintf(file, "%lf;", scores[i * diff2 + j]);
        }
        //fprintf(file, "\n");
    }
    //fclose(file);
    //free(tmp);

    // Release allocated memory
    for (i = 0; i < n ; i++)
    {
        free(disposition[i]->score);
        free(disposition[i]->x);
        free(disposition[i]->y);
        free(disposition[i]);
    }
    free(disposition);
    free(f);
    free(scores);
    freeFeatureMapObject(&map);
    return LATENT_SVM_OK;
}

/*
// Computation objective function D according the original paper
//
// API
// int filterDispositionLevel(const CvLSVMFilterObject *Fi, const featurePyramid *H,
                              int level, float **scoreFi,
                              int **pointsX, int **pointsY);
// INPUT
// Fi                - filter object (weights and coefficients of penalty
                       function that are used in this routine)
// H                 - feature pyramid
// level             - level number
// OUTPUT
// scoreFi           - values of distance transform on the level at all positions
// (pointsX, pointsY)- positions that correspond to the maximum value
                       of distance transform at all grid nodes
// RESULT
// Error status
*/
int filterDispositionLevel(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *pyramid,
                           float **scoreFi,
                           int **pointsX, int **pointsY)
{
    int n1, m1, n2, m2, /*p,*/ size, diff1, diff2;
    float *f;
    int i1, j1;
    int res;

    n1 = pyramid->sizeY;
    m1 = pyramid->sizeX;
    n2 = Fi->sizeY;
    m2 = Fi->sizeX;
    //p = pyramid->numFeatures;
    (*scoreFi) = NULL;
    (*pointsX) = NULL;
    (*pointsY) = NULL;

    // Processing the situation when part filter goes
    // beyond the boundaries of the block set
    if (n1 < n2 || m1 < m2)
    {
        return FILTER_OUT_OF_BOUNDARIES;
    } /* if (n1 < n2 || m1 < m2) */

    // Computation number of positions for the filter
    diff1 = n1 - n2 + 1;
    diff2 = m1 - m2 + 1;
    size = diff1 * diff2;

    // Allocation memory for additional array (must be free in this function)
    f = (float *)malloc(sizeof(float) * size);
    // Allocation memory for arrays for saving decisions
    (*scoreFi) = (float *)malloc(sizeof(float) * size);
    (*pointsX) = (int *)malloc(sizeof(int) * size);
    (*pointsY) = (int *)malloc(sizeof(int) * size);

    // Consruction values of the array f
    // (a dot product vectors of feature map and weights of the filter)
    res = convolution( Fi, pyramid, f );
    if (res != LATENT_SVM_OK)
    {
        free( f );
        free( *scoreFi );
        free( *pointsX );
        free( *pointsY );
        return res;
    }

    // TODO: necessary to change
    for (i1 = 0; i1 < diff1; i1++)
    {
         for (j1 = 0; j1 < diff2; j1++)
         {
             f[i1 * diff2 + j1] *= (-1);
         }
    }

    // Decision of the general distance transform task
    DistanceTransformTwoDimensionalProblem(f, diff1, diff2, Fi->fineFunction,
                                          (*scoreFi), (*pointsX), (*pointsY));

    // Release allocated memory
    free(f);
    return LATENT_SVM_OK;
}

static CvLSVMFeatureMap* featureMapBorderPartFilter(CvLSVMFeatureMap *map,
                                       int maxXBorder, int maxYBorder)
{
    int bx, by;
    int sizeX, sizeY, i, j, k;
    CvLSVMFeatureMap *new_map;

    computeBorderSize( maxXBorder, maxYBorder, &bx, &by );
    sizeX = map->sizeX + 2 * bx;
    sizeY = map->sizeY + 2 * by;
    allocFeatureMapObject(&new_map, sizeX, sizeY, map->numFeatures);
    for (i = 0; i < sizeX * sizeY * map->numFeatures; i++)
    {
        new_map->map[i] = 0.0f;
    }
    for (i = by; i < map->sizeY + by; i++)
    {
        for (j = bx; j < map->sizeX + bx; j++)
        {
            for (k = 0; k < map->numFeatures; k++)
            {
                new_map->map[(i * sizeX + j) * map->numFeatures + k] =
                    map->map[((i - by) * map->sizeX + j - bx) * map->numFeatures + k];
            }
        }
    }
    return new_map;
}

}