#ifndef _CLSVMDetector_H_
#define _CLSVMDetector_H_

#include "LibLSVM.hpp"
#include "LibPartFilter.h"
#include "LibPyramid.h"

using namespace cv;
using namespace std;

namespace liblsvm
{

class CLSVMDetector
{
public:
	struct CV_EXPORTS ObjectDetection
    {
        ObjectDetection();
        ObjectDetection( const Rect& rect, float score, int classID=-1 );
        Rect rect;
        float score;
        int classID;
    };

	CLSVMDetector();
	CLSVMDetector( const vector<string>& filenames, const vector<string>& classNames=vector<string>() );
	virtual ~CLSVMDetector();

	virtual void clear();
	virtual bool empty() const;
	bool load( const vector<string>& filenames, const vector<string>& classNames=vector<string>() );

    virtual void detect( const Mat& image,
                         vector<ObjectDetection>& objectDetections,
                         float overlapThreshold=0.5f,
                         int numThreads=-1 );

    virtual void detect_dft( const Mat& image,
                         vector<ObjectDetection>& objectDetections,
                         float overlapThreshold=0.5f,
                         int numThreads=-1 );

    const vector<string>& getClassNames() const;
    size_t getClassCount() const;

private:
	CvLatentSvmDetector* cvLoadLatentSvmDetector( const char* filename );
	void 	cvReleaseLatentSvmDetector( CvLatentSvmDetector** detector );
	CvSeq* 	cvLatentSvmDetectObjects( IplImage* image,
                                	  CvLatentSvmDetector* detector,
                                	  CvMemStorage* storage,
                                	  float overlap_threshold = 0.5f,
									  int numThreads = -1 );

private:
	vector<CvLatentSvmDetector*>	m_vDetectors;
    vector<string> 					m_vClassNames;
};

}

#endif