#include "CLSVMDetector.h"

namespace liblsvm 
{
CLSVMDetector::ObjectDetection::ObjectDetection() : score(0.f), classID(-1)
{

}

CLSVMDetector::ObjectDetection::ObjectDetection( const Rect& _rect, float _score, int _classID ) :
		rect(_rect), score(_score), classID(_classID)
{

}

CLSVMDetector::CLSVMDetector()
{

}

CLSVMDetector::CLSVMDetector( const vector<string>& filenames, const vector<string>& _classNames )
{
    load( filenames, _classNames );
}

CLSVMDetector::~CLSVMDetector()
{
	clear();
}

void CLSVMDetector::clear()
{
	for( size_t i = 0; i < m_vDetectors.size(); i++ )
	{
		cvReleaseLatentSvmDetector( &m_vDetectors[i] );
	}

	m_vDetectors.clear();
	m_vClassNames.clear();
}

bool CLSVMDetector::empty() const
{
	return m_vDetectors.empty();
}

const vector<string>& CLSVMDetector::getClassNames() const
{
	return m_vClassNames;
}

size_t CLSVMDetector::getClassCount() const
{
		return m_vClassNames.size();
}

bool CLSVMDetector::load( const vector<string>& filenames, const vector<string>& _classNames )
{
	clear();

	CV_Assert( _classNames.empty() || _classNames.size() == filenames.size() );

	for( size_t i = 0; i < filenames.size(); i++ )
	{
	    const string filename = filenames[i];
	    if( filename.length() < 5 || filename.substr(filename.length()-4, 4) != ".xml" )
	    {
	        continue;
	    }

	    CvLatentSvmDetector* detector = cvLoadLatentSvmDetector( filename.c_str() );
        cout<<detector->filters[0][0].numFeatures<<endl;
	    if( detector )
	    {
	        m_vDetectors.push_back( detector );
	        if( _classNames.empty() )
	        {
	            m_vClassNames.push_back( extractModelName(filenames[i]) );
	        }
	        else
	        {
	            m_vClassNames.push_back( _classNames[i] );
	        }
	    }
	}

	return !empty();
}

void CLSVMDetector::detect( const Mat& image,
                            vector<ObjectDetection>& objectDetections,
                            float overlapThreshold,
                            int numThreads )
{
    objectDetections.clear();
    if( numThreads <= 0 )
    {
        numThreads = 1;
    }

    for( size_t classID = 0; classID < m_vDetectors.size(); classID++ )
    {
        IplImage image_ipl = image;
        CvMemStorage* pStorage = cvCreateMemStorage(0);
        
        CvSeq* pDetections = cvLatentSvmDetectObjects( &image_ipl, m_vDetectors[classID], pStorage, overlapThreshold, numThreads );

        // convert results
        objectDetections.reserve( objectDetections.size() + pDetections->total );
        for( int detectionIdx = 0; detectionIdx < pDetections->total; detectionIdx++ )
        {
            CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( pDetections, detectionIdx );
            objectDetections.push_back( ObjectDetection(Rect(detection.rect), detection.score, (int)classID) );
        }
		
        cvReleaseMemStorage( &pStorage );
    }
}

void CLSVMDetector::detect_dft( const Mat& image,
                                vector<ObjectDetection>& objectDetections,
                                float overlapThreshold,
                                int numThreads )
{
    objectDetections.clear();
    if( numThreads <= 0 )
    {
        numThreads = 1;
    }

    for( size_t classID = 0; classID < m_vDetectors.size(); classID++ )
    {
        IplImage image_ipl = image;
        CvMemStorage* pStorage = cvCreateMemStorage(0);
        
        CvSeq* pDetections = cvLatentSvmDetectObjects( &image_ipl, m_vDetectors[classID], pStorage, overlapThreshold, numThreads );

        // convert results
        objectDetections.reserve( objectDetections.size() + pDetections->total );
        for( int detectionIdx = 0; detectionIdx < pDetections->total; detectionIdx++ )
        {
            CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( pDetections, detectionIdx );
            objectDetections.push_back( ObjectDetection(Rect(detection.rect), detection.score, (int)classID) );
        }
        
        cvReleaseMemStorage( &pStorage );
    }
}

CvSeq* CLSVMDetector::cvLatentSvmDetectObjects(	IplImage* image,
                                				CvLatentSvmDetector* detector,
                                				CvMemStorage* storage,
                                				float overlap_threshold, int numThreads)
{
	
	CvLSVMFeaturePyramid *H = 0;
    CvPoint *points = 0, *oppPoints = 0;
    int kPoints = 0;
    float *score = 0;
    unsigned int maxXBorder = 0, maxYBorder = 0;
    int numBoxesOut = 0;
    CvPoint *pointsOut = 0;
    CvPoint *oppPointsOut = 0;
    float *scoreOut = 0;
    CvSeq* result_seq = 0;
    int error = 0;

    // DFT image
    CvLSVMFftPyramid * pDftPyramid = 0;
    int nSizeX;
    int nSizeY;

    if(image->nChannels == 3)
    {
        cvCvtColor(image, image, CV_BGR2RGB);
    }

    // Getting maximum filter dimensions
    getMaxFilterDims((const CvLSVMFilterObject**)(detector->filters), detector->num_components, detector->num_part_filters, &maxXBorder, &maxYBorder);

    // Create feature pyramid with nullable border
    H = createFeaturePyramidWithBorder(image, maxXBorder, maxYBorder);
    

    // Search object
    error = searchObjectThresholdSomeComponents(H, (const CvLSVMFilterObject**)(detector->filters),
        detector->num_components, detector->num_part_filters, detector->b, detector->score_threshold,
        &points, &oppPoints, &score, &kPoints, numThreads);
    if( error != LATENT_SVM_OK )
    {
        return NULL;
    }
    // Clipping boxes
    clippingBoxes( image->width, image->height, points, kPoints );
    clippingBoxes( image->width, image->height, oppPoints, kPoints );
    // NMS procedure
    nonMaximumSuppression( kPoints, points, oppPoints, score, overlap_threshold, &numBoxesOut, &pointsOut, &oppPointsOut, &scoreOut );

    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvObjectDetection), storage );

    for( int i = 0; i < numBoxesOut; i++ )
    {
        CvObjectDetection detection = {{0, 0, 0, 0}, 0};
        detection.score = scoreOut[i];
        CvRect bounding_box = {0, 0, 0, 0};
        bounding_box.x = pointsOut[i].x;
        bounding_box.y = pointsOut[i].y;
        bounding_box.width = oppPointsOut[i].x - pointsOut[i].x;
        bounding_box.height = oppPointsOut[i].y - pointsOut[i].y;
        detection.rect = bounding_box;
        cvSeqPush(result_seq, &detection);
    }

    if(image->nChannels == 3)
        cvCvtColor(image, image, CV_RGB2BGR);

    freeFeaturePyramidObject(&H);
    free(points);
    free(oppPoints);
    free(score);

    return result_seq;
}

/*
// load trained detector from a file
//
// API
// CvLatentSvmDetector* cvLoadLatentSvmDetector(const char* filename);
// INPUT
// filename             - path to the file containing the parameters of
//                      - trained Latent SVM detector
// OUTPUT
// trained Latent SVM detector in internal representation
*/
CvLatentSvmDetector* CLSVMDetector::cvLoadLatentSvmDetector(const char* filename)
{
    CvLatentSvmDetector* detector = 0;
    CvLSVMFilterObject** filters = 0;
    int kFilters = 0;
    int kComponents = 0;
    int* kPartFilters = 0;
    float* b = 0;
    float scoreThreshold = 0.f;
    int err_code = 0;

    err_code = loadModel(filename, &filters, &kFilters, &kComponents, &kPartFilters, &b, &scoreThreshold);
    if (err_code != LATENT_SVM_OK) return 0;

    detector = (CvLatentSvmDetector*)malloc(sizeof(CvLatentSvmDetector));
    detector->filters = filters;
    detector->b = b;
    detector->num_components = kComponents;
    detector->num_filters = kFilters;
    detector->num_part_filters = kPartFilters;
    detector->score_threshold = scoreThreshold;

    return detector;
}

void CLSVMDetector::cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector)
{
    free((*detector)->b);
    free((*detector)->num_part_filters);
    for (int i = 0; i < (*detector)->num_filters; i++)
    {
        free((*detector)->filters[i]->H);
        free((*detector)->filters[i]);
    }
    free((*detector)->filters);
    free((*detector));
    *detector = 0;
}

} // liblsvm
