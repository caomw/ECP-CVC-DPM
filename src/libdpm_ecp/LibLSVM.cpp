#include "LibLSVM.hpp"

namespace liblsvm
{
string extractModelName( const string& filename )
{
    size_t startPos = filename.rfind('/');
    if( startPos == string::npos )
        startPos = filename.rfind('\\');

    if( startPos == string::npos )
        startPos = 0;
    else
        startPos++;

    const int extentionSize = 4; //.xml

    int substrLength = (int)(filename.size() - startPos - extentionSize);

    return filename.substr(startPos, substrLength);
}

IplImage* resize_opencv(IplImage* img, float scale)
{
    IplImage* imgTmp;

    int W, H, tW, tH;

    W = img->width;
    H = img->height;

    tW = (int)(((float)W) * scale + 0.5);
    tH = (int)(((float)H) * scale + 0.5);

    imgTmp = cvCreateImage(cvSize(tW , tH), img->depth, img->nChannels);
    cvResize(img, imgTmp, CV_INTER_AREA);

    return imgTmp;
}

void sort(int n, const float* x, int* indices)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
        {
            if (x[indices[j]] > x[indices[i]])
            {
                //float x_tmp = x[i];
                int index_tmp = indices[i];
                //x[i] = x[j];
                indices[i] = indices[j];
                //x[j] = x_tmp;
                indices[j] = index_tmp;
            }
        }
}

int DistanceTransformTwoDimensionalProblem(const float *f,
                                           const int n, const int m,
                                           const float coeff[4],
                                           float *distanceTransform,
                                           int *pointsX, int *pointsY)
{
    int i, j, tmp;
    int resOneDimProblem;
    int size = n * m;
    std::vector<float> internalDistTrans(size);
    std::vector<int> internalPointsX(size);

    for (i = 0; i < n; i++)
    {
        resOneDimProblem = DistanceTransformOneDimensionalProblem(
                                    f + i * m, m,
                                    coeff[0], coeff[2],
                                    &internalDistTrans[i * m],
                                    &internalPointsX[i * m]);
        if (resOneDimProblem != DISTANCE_TRANSFORM_OK)
            return DISTANCE_TRANSFORM_ERROR;
    }
    Transpose(&internalDistTrans[0], n, m);
    for (j = 0; j < m; j++)
    {
        resOneDimProblem = DistanceTransformOneDimensionalProblem(
                                    &internalDistTrans[j * n], n,
                                    coeff[1], coeff[3],
                                    distanceTransform + j * n,
                                    pointsY + j * n);
        if (resOneDimProblem != DISTANCE_TRANSFORM_OK)
            return DISTANCE_TRANSFORM_ERROR;
    }
    Transpose(distanceTransform, m, n);
    Transpose_int(pointsY, m, n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            tmp = pointsY[i * m + j];
            pointsX[i * m + j] = internalPointsX[tmp * m + j];
        }
    }

    return DISTANCE_TRANSFORM_OK;
}

/*
// Decision of one dimensional problem generalized distance transform
// on the regular grid at all points
//      min (a(y' - y) + b(y' - y)(y' - y) + f(y')) (on y')
//
// API
// int DistanceTransformOneDimensionalProblem(const float *f, const int n,
                                              const float a, const float b,
                                              float *distanceTransform,
                                              int *points);
// INPUT
// f                 - function on the regular grid
// n                 - grid dimension
// a                 - coefficient of optimizable function
// b                 - coefficient of optimizable function
// OUTPUT
// distanceTransform - values of generalized distance transform
// points            - arguments that corresponds to the optimal value of function
// RESULT
// Error status
*/
int DistanceTransformOneDimensionalProblem(const float *f, const int n,
                                           const float a, const float b,
                                           float *distanceTransform,
                                           int *points)
{
    int i, k;
    int tmp;
    int diff;
    float pointIntersection;
    int *v;
    float *z;
    k = 0;

    // Allocation memory (must be free in this function)
    v = (int *)malloc (sizeof(int) * n);
    z = (float *)malloc (sizeof(float) * (n + 1));

    v[0] = 0;
    z[0] = (float)F_MIN; // left border of envelope
    z[1] = (float)F_MAX; // right border of envelope

    for (i = 1; i < n; i++)
    {
        tmp = GetPointOfIntersection(f, a, b, v[k], i, &pointIntersection);
        if (tmp != DISTANCE_TRANSFORM_OK)
        {
            free(v);
            free(z);
            return DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR;
        } /* if (tmp != DISTANCE_TRANSFORM_OK) */
        if (pointIntersection <= z[k])
        {
            // Envelope doesn't contain current parabola
            do
            {
                k--;
                tmp = GetPointOfIntersection(f, a, b, v[k], i, &pointIntersection);
                if (tmp != DISTANCE_TRANSFORM_OK)
                {
                    free(v);
                    free(z);
                    return DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR;
                } /* if (tmp != DISTANCE_TRANSFORM_OK) */
            }while (pointIntersection <= z[k]);
            // Addition parabola to the envelope
            k++;
            v[k] = i;
            z[k] = pointIntersection;
            z[k + 1] = (float)F_MAX;
        }
        else
        {
            // Addition parabola to the envelope
            k++;
            v[k] = i;
            z[k] = pointIntersection;
            z[k + 1] = (float)F_MAX;
        } /* if (pointIntersection <= z[k]) */
    }

    // Computation values of generalized distance transform at all grid points
    k = 0;
    for (i = 0; i < n; i++)
    {
        while (z[k + 1] < i)
        {
            k++;
        }
        points[i] = v[k];
        diff = i - v[k];
        distanceTransform[i] = a * diff + b * diff * diff + f[v[k]];
    }

    // Release allocated memory
    free(v);
    free(z);
    return DISTANCE_TRANSFORM_OK;
}

/*
// Computation the point of intersection functions
// (parabolas on the variable y)
//      a(y - q1) + b(q1 - y)(q1 - y) + f[q1]
//      a(y - q2) + b(q2 - y)(q2 - y) + f[q2]
//
//
// API
// int GetPointOfIntersection(const float *f,
                              const float a, const float b,
                              int q1, int q2, float *point);
// INPUT
// f                - function on the regular grid
// a                - coefficient of the function
// b                - coefficient of the function
// q1               - parameter of the function
// q2               - parameter of the function
// OUTPUT
// point            - point of intersection
// RESULT
// Error status
*/
int GetPointOfIntersection(const float *f,
                           const float a, const float b,
                           int q1, int q2, float *point)
{
    if (q1 == q2)
    {
        return DISTANCE_TRANSFORM_EQUAL_POINTS;
    } /* if (q1 == q2) */
    (*point) = ( (f[q2] - a * q2 + b *q2 * q2) -
                 (f[q1] - a * q1 + b * q1 * q1) ) / (2 * b * (q2 - q1));
    return DISTANCE_TRANSFORM_OK;
}

/*
// Getting transposed matrix
//
// API
// void Transpose(float *a, int n, int m);
// INPUT
// a                 - initial matrix
// n                 - number of rows
// m                 - number of columns
// OUTPUT
// a                 - transposed matrix
// RESULT
// None
*/
void Transpose(float *a, int n, int m)
{
    int *cycle;
    int i, k, q, cycle_len;
    int max_cycle_len;

    max_cycle_len = n * m;

    // Allocation memory  (must be free in this function)
    cycle = (int *)malloc(sizeof(int) * max_cycle_len);

    cycle_len = 0;
    q = n * m - 1;
    for (i = 1; i < q; i++)
    {
        k = GetNextCycleElement(i, n, q);
        cycle[cycle_len] = i;
        cycle_len++;

        while (k > i)
        {
            cycle[cycle_len] = k;
            cycle_len++;
            k = GetNextCycleElement(k, n, q);
        }
        if (k == i)
        {
            TransposeCycleElements(a, cycle, cycle_len);
        } /* if (k == i) */
        cycle_len = 0;
    }

    // Release allocated memory
    free(cycle);
}

/*
// Getting transposed matrix
//
// API
// void Transpose_int(int *a, int n, int m);
// INPUT
// a                 - initial matrix
// n                 - number of rows
// m                 - number of columns
// OUTPUT
// a                 - transposed matrix
// RESULT
// None
*/
static void Transpose_int(int *a, int n, int m)
{
    int *cycle;
    int i, k, q, cycle_len;
    int max_cycle_len;

    max_cycle_len = n * m;

    // Allocation memory  (must be free in this function)
    cycle = (int *)malloc(sizeof(int) * max_cycle_len);

    cycle_len = 0;
    q = n * m - 1;
    for (i = 1; i < q; i++)
    {
        k = GetNextCycleElement(i, n, q);
        cycle[cycle_len] = i;
        cycle_len++;

        while (k > i)
        {
            cycle[cycle_len] = k;
            cycle_len++;
            k = GetNextCycleElement(k, n, q);
        }
        if (k == i)
        {
            TransposeCycleElements_int(a, cycle, cycle_len);
        } /* if (k == i) */
        cycle_len = 0;
    }

    // Release allocated memory
    free(cycle);
}

/*
// Transpose cycle elements
//
// API
// void TransposeCycleElements(float *a, int *cycle, int cycle_len)
// INPUT
// a                 - initial matrix
// cycle             - indeces array of cycle
// cycle_len         - number of elements in the cycle
// OUTPUT
// a                 - matrix with transposed elements
// RESULT
// Error status
*/
void TransposeCycleElements(float *a, int *cycle, int cycle_len)
{
    int i;
    float buf;
    for (i = cycle_len - 1; i > 0 ; i--)
    {
        buf = a[ cycle[i] ];
        a[ cycle[i] ] = a[ cycle[i - 1] ];
        a[ cycle[i - 1] ] = buf;
    }
}

/*
// Transpose cycle elements
//
// API
// void TransposeCycleElements(int *a, int *cycle, int cycle_len)
// INPUT
// a                 - initial matrix
// cycle             - indeces array of cycle
// cycle_len         - number of elements in the cycle
// OUTPUT
// a                 - matrix with transposed elements
// RESULT
// Error status
*/
static void TransposeCycleElements_int(int *a, int *cycle, int cycle_len)
{
    int i;
    int buf;
    for (i = cycle_len - 1; i > 0 ; i--)
    {
        buf = a[ cycle[i] ];
        a[ cycle[i] ] = a[ cycle[i - 1] ];
        a[ cycle[i - 1] ] = buf;
    }
}

/*
// Computation next cycle element
//
// API
// int GetNextCycleElement(int k, int n, int q);
// INPUT
// k                 - index of the previous cycle element
// n                 - number of matrix rows
// q                 - parameter that equal
                       (number_of_rows * number_of_columns - 1)
// OUTPUT
// None
// RESULT
// Next cycle element
*/
int GetNextCycleElement(int k, int n, int q)
{
    return ((k * n) % q);
}

int loadModel(
              const char *modelPath,
              CvLSVMFilterObject ***filters,
              int *kFilters,
              int *kComponents,
              int **kPartFilters,
              float **b,
              float *scoreThreshold)
{
    int last;
    int max;
    int *comp;
    int count;
    int i;
    int err;
    float score;
    //printf("start_parse\n\n");

    err = LSVMparser(modelPath, filters, &last, &max, &comp, b, &count, &score);
    if(err != LATENT_SVM_OK){
        return err;
    }
    (*kFilters)       = last + 1;
    (*kComponents)    = count;
    (*scoreThreshold) = (float) score;

    (*kPartFilters) = (int *)malloc(sizeof(int) * count);

    for(i = 1; i < count;i++){
        (*kPartFilters)[i] = (comp[i] - comp[i - 1]) - 1;
    }
    (*kPartFilters)[0] = comp[0];

    return 0;
}

int isMODEL    (char *str)
{
    char stag [] = "<Model>";
    char etag [] = "</Model>";
    if(strcmp(stag, str) == 0)return  MODEL;
    if(strcmp(etag, str) == 0)return EMODEL;
    return 0;
}
int isP        (char *str)
{
    char stag [] = "<P>";
    char etag [] = "</P>";
    if(strcmp(stag, str) == 0)return  P;
    if(strcmp(etag, str) == 0)return EP;
    return 0;
}
int isSCORE        (char *str)
{
    char stag [] = "<ScoreThreshold>";
    char etag [] = "</ScoreThreshold>";
    if(strcmp(stag, str) == 0)return  SCORE;
    if(strcmp(etag, str) == 0)return ESCORE;
    return 0;
}
int isCOMP     (char *str)
{
    char stag [] = "<Component>";
    char etag [] = "</Component>";
    if(strcmp(stag, str) == 0)return  COMP;
    if(strcmp(etag, str) == 0)return ECOMP;
    return 0;
}
int isRFILTER  (char *str)
{
    char stag [] = "<RootFilter>";
    char etag [] = "</RootFilter>";
    if(strcmp(stag, str) == 0)return  RFILTER;
    if(strcmp(etag, str) == 0)return ERFILTER;
    return 0;
}
int isPFILTERs (char *str)
{
    char stag [] = "<PartFilters>";
    char etag [] = "</PartFilters>";
    if(strcmp(stag, str) == 0)return  PFILTERs;
    if(strcmp(etag, str) == 0)return EPFILTERs;
    return 0;
}

int isPFILTER  (char *str)
{
    char stag [] = "<PartFilter>";
    char etag [] = "</PartFilter>";
    if(strcmp(stag, str) == 0)return  PFILTER;
    if(strcmp(etag, str) == 0)return EPFILTER;
    return 0;
}

int isSIZEX    (char *str)
{
    char stag [] = "<sizeX>";
    char etag [] = "</sizeX>";
    if(strcmp(stag, str) == 0)return  SIZEX;
    if(strcmp(etag, str) == 0)return ESIZEX;
    return 0;
}
int isSIZEY    (char *str)
{
    char stag [] = "<sizeY>";
    char etag [] = "</sizeY>";
    if(strcmp(stag, str) == 0)return  SIZEY;
    if(strcmp(etag, str) == 0)return ESIZEY;
    return 0;
}
int isWEIGHTS  (char *str)
{
    char stag [] = "<Weights>";
    char etag [] = "</Weights>";
    if(strcmp(stag, str) == 0)return  WEIGHTS;
    if(strcmp(etag, str) == 0)return EWEIGHTS;
    return 0;
}
int isV        (char *str)
{
    char stag [] = "<V>";
    char etag [] = "</V>";
    if(strcmp(stag, str) == 0)return  TAGV;
    if(strcmp(etag, str) == 0)return ETAGV;
    return 0;
}
int isVx       (char *str)
{
    char stag [] = "<Vx>";
    char etag [] = "</Vx>";
    if(strcmp(stag, str) == 0)return  Vx;
    if(strcmp(etag, str) == 0)return EVx;
    return 0;
}
int isVy       (char *str)
{
    char stag [] = "<Vy>";
    char etag [] = "</Vy>";
    if(strcmp(stag, str) == 0)return  Vy;
    if(strcmp(etag, str) == 0)return EVy;
    return 0;
}
int isD        (char *str)
{
    char stag [] = "<Penalty>";
    char etag [] = "</Penalty>";
    if(strcmp(stag, str) == 0)return  TAGD;
    if(strcmp(etag, str) == 0)return ETAGD;
    return 0;
}
int isDx       (char *str)
{
    char stag [] = "<dx>";
    char etag [] = "</dx>";
    if(strcmp(stag, str) == 0)return  Dx;
    if(strcmp(etag, str) == 0)return EDx;
    return 0;
}
int isDy       (char *str)
{
    char stag [] = "<dy>";
    char etag [] = "</dy>";
    if(strcmp(stag, str) == 0)return  Dy;
    if(strcmp(etag, str) == 0)return EDy;
    return 0;
}
int isDxx      (char *str)
{
    char stag [] = "<dxx>";
    char etag [] = "</dxx>";
    if(strcmp(stag, str) == 0)return  Dxx;
    if(strcmp(etag, str) == 0)return EDxx;
    return 0;
}
int isDyy      (char *str)
{
    char stag [] = "<dyy>";
    char etag [] = "</dyy>";
    if(strcmp(stag, str) == 0)return  Dyy;
    if(strcmp(etag, str) == 0)return EDyy;
    return 0;
}
int isB      (char *str)
{
    char stag [] = "<LinearTerm>";
    char etag [] = "</LinearTerm>";
    if(strcmp(stag, str) == 0)return  BTAG;
    if(strcmp(etag, str) == 0)return EBTAG;
    return 0;
}

int getTeg(char *str)
{
    int sum = 0;
    sum = isMODEL (str)+
    isP        (str)+
    isSCORE    (str)+
    isCOMP     (str)+
    isRFILTER  (str)+
    isPFILTERs (str)+
    isPFILTER  (str)+
    isSIZEX    (str)+
    isSIZEY    (str)+
    isWEIGHTS  (str)+
    isV        (str)+
    isVx       (str)+
    isVy       (str)+
    isD        (str)+
    isDx       (str)+
    isDy       (str)+
    isDxx      (str)+
    isDyy      (str)+
    isB        (str);

    return sum;
}

void addFilter(CvLSVMFilterObject *** model, int *last, int *max)
{
    CvLSVMFilterObject ** nmodel;
    int i;
    (*last) ++;
    if((*last) >= (*max)){
        (*max) += 10;
        nmodel = (CvLSVMFilterObject **)malloc(sizeof(CvLSVMFilterObject *) * (*max));
        for(i = 0; i < *last; i++){
            nmodel[i] = (* model)[i];
        }
        free(* model);
        (*model) = nmodel;
    }
    (*model) [(*last)] = (CvLSVMFilterObject *)malloc(sizeof(CvLSVMFilterObject));
}

void parserRFilter  (FILE * xmlf, int p, CvLSVMFilterObject * model, float *b)
{
    int st = 0;
    int sizeX=0, sizeY=0;
    int tag;
    int tagVal;
    char ch;
    int i,j,ii;
    char buf[1024];
    char tagBuf[1024];
    double *data;
    //printf("<RootFilter>\n");

    model->V.x = 0;
    model->V.y = 0;
    model->V.l = 0;
    model->fineFunction[0] = 0.0;
    model->fineFunction[1] = 0.0;
    model->fineFunction[2] = 0.0;
    model->fineFunction[3] = 0.0;

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';

                tagVal = getTeg(tagBuf);

                if(tagVal == ERFILTER){
                    //printf("</RootFilter>\n");
                    return;
                }
                if(tagVal == SIZEX){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEX){
                    st = 0;
                    buf[i] = '\0';
                    sizeX = atoi(buf);
                    model->sizeX = sizeX;
                    //printf("<sizeX>%d</sizeX>\n", sizeX);
                }
                if(tagVal == SIZEY){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEY){
                    st = 0;
                    buf[i] = '\0';
                    sizeY = atoi(buf);
                    model->sizeY = sizeY;
                    //printf("<sizeY>%d</sizeY>\n", sizeY);
                }
                if(tagVal == WEIGHTS){
                    data = (double *)malloc( sizeof(double) * p * sizeX * sizeY);
                    size_t elements_read = fread(data, sizeof(double), p * sizeX * sizeY, xmlf);
                    CV_Assert(elements_read == (size_t)(p * sizeX * sizeY));
                    model->H = (float *)malloc(sizeof(float)* p * sizeX * sizeY);
                    for(ii = 0; ii < p * sizeX * sizeY; ii++){
                        model->H[ii] = (float)data[ii];
                    }
                    free(data);
                }
                if(tagVal == EWEIGHTS){
                    //printf("WEIGHTS OK\n");
                }
                if(tagVal == BTAG){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EBTAG){
                    st = 0;
                    buf[i] = '\0';
                    *b =(float) atof(buf);
                    //printf("<B>%f</B>\n", *b);
                }

                tag = 0;
                i   = 0;
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }
}

void parserV  (FILE * xmlf, int /*p*/, CvLSVMFilterObject * model)
{
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("    <V>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';

                tagVal = getTeg(tagBuf);

                if(tagVal == ETAGV){
                    //printf("    </V>\n");
                    return;
                }
                if(tagVal == Vx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EVx){
                    st = 0;
                    buf[i] = '\0';
                    model->V.x = atoi(buf);
                    //printf("        <Vx>%d</Vx>\n", model->V.x);
                }
                if(tagVal == Vy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EVy){
                    st = 0;
                    buf[i] = '\0';
                    model->V.y = atoi(buf);
                    //printf("        <Vy>%d</Vy>\n", model->V.y);
                }
                tag = 0;
                i   = 0;
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }
}
void parserD  (FILE * xmlf, int /*p*/, CvLSVMFilterObject * model)
{
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("    <D>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';

                tagVal = getTeg(tagBuf);

                if(tagVal == ETAGD){
                    //printf("    </D>\n");
                    return;
                }
                if(tagVal == Dx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDx){
                    st = 0;
                    buf[i] = '\0';

                    model->fineFunction[0] = (float)atof(buf);
                    //printf("        <Dx>%f</Dx>\n", model->fineFunction[0]);
                }
                if(tagVal == Dy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDy){
                    st = 0;
                    buf[i] = '\0';

                    model->fineFunction[1] = (float)atof(buf);
                    //printf("        <Dy>%f</Dy>\n", model->fineFunction[1]);
                }
                if(tagVal == Dxx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDxx){
                    st = 0;
                    buf[i] = '\0';

                    model->fineFunction[2] = (float)atof(buf);
                    //printf("        <Dxx>%f</Dxx>\n", model->fineFunction[2]);
                }
                if(tagVal == Dyy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDyy){
                    st = 0;
                    buf[i] = '\0';

                    model->fineFunction[3] = (float)atof(buf);
                    //printf("        <Dyy>%f</Dyy>\n", model->fineFunction[3]);
                }

                tag = 0;
                i   = 0;
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }
}

void parserPFilter  (FILE * xmlf, int p, int /*N_path*/, CvLSVMFilterObject * model)
{
    int st = 0;
    int sizeX=0, sizeY=0;
    int tag;
    int tagVal;
    char ch;
    int i,j, ii;
    char buf[1024];
    char tagBuf[1024];
    double *data;
    //printf("<PathFilter> (%d)\n", N_path);

    model->V.x = 0;
    model->V.y = 0;
    model->V.l = 0;
    model->fineFunction[0] = 0.0f;
    model->fineFunction[1] = 0.0f;
    model->fineFunction[2] = 0.0f;
    model->fineFunction[3] = 0.0f;

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';

                tagVal = getTeg(tagBuf);

                if(tagVal == EPFILTER){
                    //printf("</PathFilter>\n");
                    return;
                }

                if(tagVal == TAGV){
                    parserV(xmlf, p, model);
                }
                if(tagVal == TAGD){
                    parserD(xmlf, p, model);
                }
                if(tagVal == SIZEX){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEX){
                    st = 0;
                    buf[i] = '\0';
                    sizeX = atoi(buf);
                    model->sizeX = sizeX;
                    //printf("<sizeX>%d</sizeX>\n", sizeX);
                }
                if(tagVal == SIZEY){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEY){
                    st = 0;
                    buf[i] = '\0';
                    sizeY = atoi(buf);
                    model->sizeY = sizeY;
                    //printf("<sizeY>%d</sizeY>\n", sizeY);
                }
                if(tagVal == WEIGHTS){
                    data = (double *)malloc( sizeof(double) * p * sizeX * sizeY);
                    size_t elements_read = fread(data, sizeof(double), p * sizeX * sizeY, xmlf);
                    CV_Assert(elements_read == (size_t)(p * sizeX * sizeY));
                    model->H = (float *)malloc(sizeof(float)* p * sizeX * sizeY);
                    for(ii = 0; ii < p * sizeX * sizeY; ii++){
                        model->H[ii] = (float)data[ii];
                    }
                    free(data);
                }
                if(tagVal == EWEIGHTS){
                    //printf("WEIGHTS OK\n");
                }
                tag = 0;
                i   = 0;
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }
}
void parserPFilterS (FILE * xmlf, int p, CvLSVMFilterObject *** model, int *last, int *max)
{
    int st = 0;
    int N_path = 0;
    int tag;
    int tagVal;
    char ch;
    int /*i,*/j;
    //char buf[1024];
    char tagBuf[1024];
    //printf("<PartFilters>\n");

    //i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';

                tagVal = getTeg(tagBuf);

                if(tagVal == EPFILTERs){
                    //printf("</PartFilters>\n");
                    return;
                }
                if(tagVal == PFILTER){
                    addFilter(model, last, max);
                    parserPFilter  (xmlf, p, N_path, (*model)[*last]);
                    N_path++;
                }
                tag = 0;
                //i   = 0;
            }else{
                if((tag == 0)&& (st == 1)){
                    //buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }
}
void parserComp (FILE * xmlf, int p, int *N_comp, CvLSVMFilterObject *** model, float *b, int *last, int *max)
{
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int /*i,*/j;
    //char buf[1024];
    char tagBuf[1024];
    //printf("<Component> %d\n", *N_comp);

    //i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';

                tagVal = getTeg(tagBuf);

                if(tagVal == ECOMP){
                    (*N_comp) ++;
                    return;
                }
                if(tagVal == RFILTER){
                    addFilter(model, last, max);
                    parserRFilter   (xmlf, p, (*model)[*last],b);
                }
                if(tagVal == PFILTERs){
                    parserPFilterS  (xmlf, p, model, last, max);
                }
                tag = 0;
                //i   = 0;
            }else{
                if((tag == 0)&& (st == 1)){
                    //buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }
}

void parserModel(FILE * xmlf, CvLSVMFilterObject *** model, int *last, int *max, int **comp, float **b, int *count, float * score)
{
    int p = 0;
    int N_comp = 0;
    int * cmp;
    float *bb;
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j, ii = 0;
    char buf[1024];
    char tagBuf[1024];

    //printf("<Model>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';

                tagVal = getTeg(tagBuf);

                if(tagVal == EMODEL){
                    //printf("</Model>\n");
                    for(ii = 0; ii <= *last; ii++){
                        (*model)[ii]->numFeatures = p;
                    }
                    * count = N_comp;
                    return;
                }
                if(tagVal == COMP){
                    if(N_comp == 0){
                        cmp = (int    *)malloc(sizeof(int));
                        bb  = (float *)malloc(sizeof(float));
                        * comp = cmp;
                        * b    = bb;
                        * count = N_comp + 1;
                    } else {
                        cmp = (int   *)malloc(sizeof(int)   * (N_comp + 1));
                        bb  = (float *)malloc(sizeof(float) * (N_comp + 1));
                        for(ii = 0; ii < N_comp; ii++){
                            cmp[ii] = (* comp)[ii];
                            bb [ii] = (* b   )[ii];
                        }
                        free(* comp);
                        free(* b   );
                        * comp = cmp;
                        * b    = bb;
                        * count = N_comp + 1;
                    }
                    parserComp(xmlf, p, &N_comp, model, &((*b)[N_comp]), last, max);
                    cmp[N_comp - 1] = *last;
                }
                if(tagVal == P){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EP){
                    st = 0;
                    buf[i] = '\0';
                    p = atoi(buf);
                    //printf("<P>%d</P>\n", p);
                }
                if(tagVal == SCORE){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESCORE){
                    st = 0;
                    buf[i] = '\0';
                    *score = (float)atof(buf);
                    //printf("<ScoreThreshold>%f</ScoreThreshold>\n", score);
                }
                tag = 0;
                i   = 0;
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }
}

int LSVMparser(const char * filename, CvLSVMFilterObject *** model, int *last, int *max, int **comp, float **b, int *count, float * score)
{
    //int st = 0;
    int tag;
    char ch;
    int /*i,*/j;
    FILE *xmlf;
    //char buf[1024];
    char tagBuf[1024];

    (*max) = 10;
    (*last) = -1;
    (*model) = (CvLSVMFilterObject ** )malloc((sizeof(CvLSVMFilterObject * )) * (*max));

    //printf("parse : %s\n", filename);

    xmlf = fopen(filename, "rb");
    if(xmlf == NULL)
        return LSVM_PARSER_FILE_NOT_FOUND;

    //i   = 0;
    j   = 0;
    //st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tag = 0;
                //i   = 0;
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                if(getTeg(tagBuf) == MODEL){
                    parserModel(xmlf, model, last, max, comp, b, count, score);
                }
            }else{
                if(tag == 0){
                    //buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }
    }

    fclose(xmlf);
    return LATENT_SVM_OK;
}

}